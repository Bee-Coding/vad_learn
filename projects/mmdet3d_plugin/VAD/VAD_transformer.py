import torch
import numpy as np
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.utils import ext_loader
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from torchvision.transforms.functional import rotate
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.VAD.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.VAD.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.VAD.modules.spatial_cross_attention import MSDeformableAttention3D


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse                       sigmoid = 1/(1+e^(-x))       inverse_sigmoid = log(y/1-y)
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapDetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.                 是否返回中间输出
        coder_norm_cfg (dict): Config of last normalization layer. Default:                 最后一个规范化层的配置。默认值：`LN`
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapDetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        "迭代式的边界框细化（iterative bounding box refinement）" 
        """
            关键设计思想
                迭代细化：每层解码器都细化参考点，逐渐接近真实位置
                残差连接：参考点更新采用残差形式：新点 = 旧点 + Δ
                梯度截断：使用.detach()防止梯度通过参考点反向传播过多层
                归一化：通过sigmoid确保参考点始终在[0,1]范围内
        """

        """Forward function for `Detr3DTransformerDecoder`.     Detr3DTransformer解码器的前向传播函数
        Args:
            query (Tensor): Input query with shape              输入q形状(num_query, bs, embed_dims)    bs： batch_size, embed_dims:特征维度
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference            ref_points: stage_1: 输入形状(bs, num_query, 2) [center_x, center_y]
                points of offset. has shape                                 stage_2: 输入形状(bs, num_query, 4) [center_x, center_y, width, height] 或 3D: [x, y, z, depth]
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for         用于细化回归结果。只有当with_box_refine为True时才会传递，否则将传递“None”。
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when      返回中间变量为false时，shape为[1, num_query, bs, embed_dims]
                return_intermediate is `False`, otherwise it has shape          否则[num_layers, num_query, bs, embed_dims]
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):                           # 在遍历可迭代对象时同时获取索引和值。
            # reference_points通常是归一化的边界框坐标，形状为：[batch_size, num_queries, 4]， 4个值通常表示：(center_x, center_y, width, height)
            reference_points_input = reference_points[..., :2].unsqueeze(   # ...：省略号，表示所有前面的维度
                2)  # BS NUM_QUERY NUM_LEVEL 2      增加一个维度是为了与多尺度特征对齐
            output = layer(                                 # output形状: [num_query, bs, embed_dims]
                output,                                     # 当前层输入
                *args,
                reference_points=reference_points_input,    # 参考点
                key_padding_mask=key_padding_mask,          # 填充掩码
                **kwargs)
            output = output.permute(1, 0, 2)                                # 交换第1维度和第0维度, 维度置换（为回归分支准备）

            if reg_branches is not None:
                tmp = reg_branches[lid](output)             # 当前层的回归分支 tmp形状：[bs, num_query, 回归输出维度]

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                # 公式: new_ref = sigmoid(inverse_sigmoid(old_ref) + delta)                                     # 参考点更新公式：
                new_reference_points[..., :2] = tmp[                                                           # 为什么要用inverse_sigmoid？
                    ..., :2] + inverse_sigmoid(reference_points[..., :2]) # inverse_sigmoid：sigmoid的逆函数。    1. old_ref是sigmoid后的值（0-1）
                # new_reference_points[..., 2:3] = tmp[                                                         2. inverse_sigmoid将其映射回实数域
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])                                   3. 加上回归分支预测的delta
                #                                                                                               4. 再次sigmoid得到新的归一化坐标
                new_reference_points = new_reference_points.sigmoid()                                          # 这样确保参考点始终在[0,1]范围内

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)                                                # 恢复标准Transformer格式    [bs, num_query, embed_dims] -> [num_query, bs, embed_dims]
            if self.return_intermediate:
                intermediate.append(output)                                 # 存储当前层输出
                intermediate_reference_points.append(reference_points)      # 存储当前层参考点

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(      # intermediate形状: [num_layers, num_query, bs, embed_dims]
                intermediate_reference_points)                  # [num_layers, bs, num_query, 2]

        # 否则只返回最后一层的结果
        return output, reference_points     # [num_query, bs, embed_dims], [bs, num_query, 2]


@TRANSFORMER.register_module()
class VADPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.                                   VAD感知Transformer - 核心感知模块
    Args:
        as_two_stage (bool): Generate query from encoder features.          是否从编码器特征生成query。
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:          FPN中的特征图数量   FPN：特征金字塔网络，
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,                                      # FPN特征金字塔层数
                 num_cams=6,                                                # 相机数量（nuScenes为6个）
                 two_stage_num_proposals=300,                               # 两阶段检测的提议数量
                 encoder=None,                                              # BEV编码器配置
                 decoder=None,                                              # 智能体解码器配置
                 map_decoder=None,                                          # 地图解码器配置
                 embed_dims=256,                                            # 特征维度
                 rotate_prev_bev=True,                                      # 是否旋转历史BEV特征
                 use_shift=True,                                            # 是否使用平移（考虑车辆运动）
                 use_can_bus=True,                                          # 是否使用CAN总线信号
                 can_bus_norm=True,                                         # 是否对CAN总线信号归一化
                 use_cams_embeds=True,                                      # 是否使用相机位置编码
                 rotate_center=[100, 100],                                  # BEV旋转中心
                 map_num_vec=50,                                            # 地图向量数量**
                 map_num_pts_per_vec=10,                                    # 每个地图向量的点数**
                 **kwargs):
        super(VADPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            self.decoder = None
        if map_decoder is not None:
            self.map_decoder = build_transformer_layer_sequence(map_decoder)
        else:
            self.map_decoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        # 1. 尺度嵌入（多尺度特征融合）
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))                             # 形状: [4, 256] - 对应FPN的4个尺度
        # 2. 相机嵌入（区分不同相机）
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))                          # 形状: [6, 256] - 对应6个相机
        # 3. 智能体参考点预测器
        self.reference_points = nn.Linear(self.embed_dims, 3)                      # 输入: 256维查询 → 输出: 3维坐标 (x, y, z或x, y, depth)
        # 4. 地图参考点预测器
        self.map_reference_points = nn.Linear(self.embed_dims, 2)                  # 输入: 256维查询 → 输出: 2维坐标 (x, y)
        # 5. CAN总线信号编码器
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),                                                  # 两个全连接层
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))      # 如果需要归一化，则再加一个归一化层

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.map_reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,                     # 多尺度特征列表 [4个尺度]
            bev_queries,                    # BEV查询 [bev_h*bev_w, 256]
            bev_h,
            bev_w,                          # BEV空间分辨率 bev_h*bev_w (如200×200)
            grid_length=[0.512, 0.512],     # 每个BEV网格的物理尺寸
            bev_pos=None,                   # BEV位置编码
            prev_bev=None,                  # 上一时刻BEV特征（时序融合）
            **kwargs):
        """
        obtain bev features.            将多视角图像特征转换为BEV特征。
        """

        bs = mlvl_feats[0].size(0)                                  # 批次大小batch_size
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)     # [bev_h*bev_w, 1, 256] → [bev_h*bev_w, bs, 256]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)               # [bev_h*bev_w, bs, 256]

        # obtain rotation angle and shift with ego motion
        # 从CAN总线信号获取车辆运动信息
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])        # x方向位移
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])        # y方向位移
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])    # 车辆朝向
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        # 计算BEV网格的平移量（考虑车辆运动）
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            # 历史BEV特征对齐（时序融合）
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]      # 旋转角度
                    tmp_prev_bev = prev_bev[:, i].reshape(                      # 旋转历史BEV特征以对齐当前帧
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals       融合CAN总线信号
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]                             # [1, bs, 256]
        bev_queries = bev_queries + can_bus * self.use_can_bus                      # 将车辆状态信息注入查询

        # 多尺度特征处理
        feat_flatten = []                                   # 存储展平后的特征
        spatial_shapes = []                                 # 存储每个尺度的空间形状
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape               # [bs, 6, 256, H, W]
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)      # [6, bs, H*W, 256]
            # 添加相机嵌入和尺度嵌入
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # 通过编码器生成BEV特征
        bev_embed = self.encoder(
            bev_queries,                            # BEV查询 [bev_h*bev_w, bs, 256]
            feat_flatten,                           # 多尺度特征 [6, H*W, bs, 256]
            feat_flatten,                           # 作为key和value
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,                        # BEV位置编码
            spatial_shapes=spatial_shapes,          # 每个尺度的空间形状
            level_start_index=level_start_index,    # 每个尺度的起始索引
            prev_bev=prev_bev,                      # 历史BEV特征
            shift=shift,                            # 平移量
            **kwargs
        )

        return bev_embed

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,                                         # 多尺度特征
                bev_queries,                                        # BEV查询
                object_query_embed,                                 # 智能体查询嵌入 [num_queries, 512]
                map_query_embed,                                    # 地图查询嵌入 [num_map_queries, 512]
                bev_h,                                              # BEV分辨率
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,                                  # 智能体回归分支
                cls_branches=None,                                  # 智能体分类分支
                map_reg_branches=None,                              # 地图回归分支
                map_cls_branches=None,                              # 地图分类分支          
                prev_bev=None,            
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        # 提取BEV特征
        bev_embed = self.get_bev_features(                          # 输出: [bs, bev_h*bev_w, 256]
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        # 智能体查询处理
        bs = mlvl_feats[0].size(0)
        # 拆分查询嵌入为 位置编码 和 查询向量
        query_pos, query = torch.split(                         #object_query_embed: [num_queries, 512] = [256] + [256]
            object_query_embed, self.embed_dims, dim=1)         # query_pos: [num_queries, 256] - 查询位置编码, # query: [num_queries, 256] - 查询向量
        # 扩展到批次维度
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)   # [bs, num_queries, 256]
        query = query.unsqueeze(0).expand(bs, -1, -1)           # [bs, num_queries, 256]
        # 生成初始参考点（3D坐标）
        reference_points = self.reference_points(query_pos)     # [bs, num_queries, 3]      3是怎么来的？3：[x,y,z]
        reference_points = reference_points.sigmoid()           # 归一化到[0,1]
        init_reference_out = reference_points

        # 拆分地图查询嵌入为 位置编码 和 查询向量   
        map_query_pos, map_query = torch.split(                         # map_query_embed: [num_queries, 512] = [256] + [256]
            map_query_embed, self.embed_dims, dim=1)                    # map_query_pos: [num_queries, 256] - 查询位置编码, # map_query: [num_queries, 256] - 查询向量
        map_query_pos = map_query_pos.unsqueeze(0).expand(bs, -1, -1)   # [bs, num_queries, 256]
        map_query = map_query.unsqueeze(0).expand(bs, -1, -1)           # [bs, num_queries, 256]
        map_reference_points = self.map_reference_points(map_query_pos) # [bs, num_map_queries, 2]  2:[x,y]
        map_reference_points = map_reference_points.sigmoid()
        map_init_reference_out = map_reference_points        

        # 调整维度顺序（适配Transformer） 从 [bs, num_queries, 256] → [num_queries, bs, 256] (Transformer标准格式)
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        map_query = map_query.permute(1, 0, 2)
        map_query_pos = map_query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        # 智能体解码（检测交通参与者）
        if self.decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            inter_states, inter_references = self.decoder(
                query=query,                                # 智能体查询 [num_queries, bs, 256]
                key=None,                                   # 不使用额外的key
                value=bev_embed,                            # BEV特征作为value [bev_h*bev_w, bs, 256]
                query_pos=query_pos,                        # 查询位置编码
                reference_points=reference_points,          # 初始参考点
                reg_branches=reg_branches,                  # 回归分支（边界框细化）
                cls_branches=cls_branches,                  # 分类分支
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),                       # device=query.device表示张量的存储位置
                **kwargs)
            inter_references_out = inter_references
        else:
            inter_states = query.unsqueeze(0)                               # 如果没有解码器，直接返回查询
            inter_references_out = reference_points.unsqueeze(0)

        # 地图解码（检测地图元素）
        if self.map_decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]    L代表解码器层的数量即num_layers
            map_inter_states, map_inter_references = self.map_decoder(
                query=map_query,            # 地图查询
                key=None,                   # 不使用额外的key
                value=bev_embed,            # 共享BEV特征作为value [bev_h*bev_w, bs, 256]
                query_pos=map_query_pos,    # 地图查询位置编码
                reference_points=map_reference_points,      # 地图初始参考点
                reg_branches=map_reg_branches,              # 地图点回归
                cls_branches=map_cls_branches,              # 地图元素分类
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=map_query.device),
                level_start_index=torch.tensor([0], device=map_query.device),
                **kwargs)
            map_inter_references_out = map_inter_references
        else:
            map_inter_states = map_query.unsqueeze(0)
            map_inter_references_out = map_reference_points.unsqueeze(0)

        # 返回所有结果
        return (
            bev_embed,                      # BEV特征 [bev_h*bev_w, bs, 256]
            inter_states,                   # 智能体解码器中间状态 [L, num_queries, bs, 256]
            init_reference_out,             # 智能体初始参考点 [bs, num_queries, 3]
            inter_references_out,           # 智能体中间参考点 [L, bs, num_queries, 3]
            map_inter_states,               # 地图解码器中间状态 [L, num_map_queries, bs, 256]
            map_init_reference_out,         # 地图初始参考点 [bs, num_map_queries, 2]
            map_inter_references_out)       # 地图中间参考点 [L, bs, num_map_queries, 2]
    # 输出: BEV特征、智能体检测结果 (状态 + 参考点)、地图检测结果 (状态 + 参考点)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.                 是否返回中间输出
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.           最后一个规范化层的配置。默认值：`LN`
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                *args,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs)

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query