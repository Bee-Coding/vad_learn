
DETR:
{
    DETR（Detection Transformer）是Facebook AI提出的目标检测模型。
    
    DETR 框架概述：图像输入 → CNN 骨干网络 → Transformer 编码器-解码器 → DETRHead → 预测结果

    在DETR中，模型主要由三个部分组成：
        1. 卷积神经网络（CNN）骨干网络，用于提取图像特征。
        2. Transformer编码器-解码器结构，用于处理图像特征并生成一组对象查询（object queries）的嵌入。
        3. 检测头（DETRHead），将解码器的输出转换为边界框坐标和类别标签。

    """Implements the DETR transformer head.                                    
    
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.       不包括背景的类别数
        in_channels (int): Number of channels in the input feature map.         输入特征图中的通道数
        num_query (int): Number of query in Transformer.                        Transformer中的查询数 （什么意思？）
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.       在'FFN'中使用的完全连接的层数，然后用于回归头。默认值2。
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.       transformer的配置
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of           是否同步所有列的avg_factor。默认为False。
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):                       位置编码的配置
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the                    分类损失的配置。默认值“L1Loss”。
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the                   回归损失的配置。默认值“L1Loss”。
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the                    配置回归iou-loss。默认值为“GIoULoss”。
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of               transformer训练配置。
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of                transformer测试配置。
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.    初始化配置dict。默认值：无
            Default: None
    """
}