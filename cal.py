import torch
from mmengine.runner import load_checkpoint
from mmengine.config import Config
from mmdet.registry import MODELS
from mmengine.analysis import get_model_complexity_info
from mmdet.structures import DetDataSample
from thop import profile

# 确保所有模型正确注册
import mmdet.models  # 触发 mmdet 模型注册
from mmdet.models.detectors import DeformableDETR  # 确保 DeformableDETR 被注册


from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.registry import MODELS

from mmdet.models.detectors.deformable_detr import DeformableDETR
MODELS.register_module(module=DeformableDETR)

from mmdet.models.detectors.deformable_detr_kd import DeformableDETRWithDistillation
MODELS.register_module(module=DeformableDETRWithDistillation)

# 手动注册 DetDataPreprocessor
MODELS.register_module(module=DetDataPreprocessor)

# 你的模型配置和权重文件
model_config_path = "work_dirs/deformable-detr_r50_16xb2-50e_coco_old/deformable-detr_r50_16xb2-50e_coco.py"
# model_checkpoint = "work_dirs/deformable-detr_r50_16xb2-50e_coco/仅有kl散度.pth"

# 加载模型配置
cfg = Config.fromfile(model_config_path)

# 确保 DeformableDETR 被正确注册
assert 'DeformableDETR' in MODELS.module_dict, "DeformableDETR 未正确注册！"

# 构造教师模型
teacher_model = MODELS.build(cfg.model)
teacher_model.to("cpu")  # 移动到 CPU（可改成 "cuda"）

# 加载权重
# load_checkpoint(teacher_model, model_checkpoint, map_location="cpu",strict=False)

# 计算总参数量
total_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params / 1e6:.2f}M")  # 以百万（M）为单位



# dummy_input = torch.randn(1, 3, 224, 224)
# flops, params = profile(teacher_model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

