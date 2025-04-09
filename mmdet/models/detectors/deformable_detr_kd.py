from mmdet.models.detectors.deformable_detr import DeformableDETR
from mmdet.registry import MODELS
from mmdet.models.losses.kd_loss import KnowledgeDistillationKLDivLoss,knowledge_distillation_kl_div_loss
import torch
import torch.nn as nn
from mmengine.config import ConfigDict
from mmengine.runner.checkpoint import load_checkpoint  # 用于加载权重
import torch.nn.functional as F
import importlib


class TeacherModel(DeformableDETR):
        def __init__(self, model_config, model_checkpoint, device=None, **kwargs):
            
            # 将 share_pred_layer, num_pred_layer, as_two_stage 从 bbox_head 移除
            bbox_head_config = kwargs['bbox_head']
            for key in ['share_pred_layer', 'num_pred_layer', 'as_two_stage']:
                if key in bbox_head_config:
                    del bbox_head_config[key]  # 使用 del 删除键，而不是 pop 并添加到 kwargs
            
            super().__init__(**kwargs)
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 直接用配置文件初始化模型
            if model_config:
                self.model = MODELS.build(model_config).to(self.device)
            else:
                self.model = None
                
            # 将模型切换为评估模式
            self.model.eval()

            # 加载教师模型的预训练权重
            if model_checkpoint:
                load_checkpoint(self.model, model_checkpoint, map_location=self.device)

            print("-----------------------------")

        def forward(self, *args, **kwargs):
            """
            使用教师模型的父类 forward 方法来获取输出。
            """
            # 调用父类的 forward 方法
            return super(TeacherModel, self).forward(*args, **kwargs)
    
@MODELS.register_module()
class DeformableDETRWithDistillation(DeformableDETR):
    
    def __init__(self, teacher_model=None, teacher_model_load_from=None, distillation_loss_weight=2, T=10, **kwargs):
        super().__init__(**kwargs)

        self.distillation_loss_weight = distillation_loss_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = T
        print(f"teacher_model: {teacher_model}")
        print(f"teacher_model_load_from: {teacher_model_load_from}")
        print("0")

        self.teacher_model = TeacherModel(teacher_model, teacher_model_load_from, self.device,**kwargs)
        print("----------------------------------------------------------------")
        print("----------------------------------------------------------------")
            

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(self, *args, **kwargs):
        # 获取学生模型的输出
        student_output = super(DeformableDETRWithDistillation, self).forward(*args, **kwargs)
        # print(student_output)  # 查看输出的所有字段
        
#         # 初始化损失
#         distillation_loss = 0
        
#         if self.teacher_model is not None:
#             with torch.no_grad():  # 不计算梯度
#                 teacher_output = self.teacher_model(*args, **kwargs)
#                 # print(teacher_output)
            
#             # 获取学生模型和教师模型的logits
#             student_logits = student_output['logits']
#             teacher_logits = teacher_output['logits']

#            # 确保logits的形状一致
#             assert student_logits.shape == teacher_logits.shape, "Shape mismatch between student and teacher logits"

#             # 使用知识蒸馏损失计算KL散度
#             distillation_loss = knowledge_distillation_kl_div_loss(
#                 student_logits, teacher_logits, T=self.T)
            
#             student_output['distillation_loss'] = distillation_loss
        
#         # 得到教师模型和学生模型的feature map
#         # 1. 得到kwargs中的batch_inputs , 也就是kwargs的inputs
#         batch_inputs = kwargs["inputs"]
        
#         # 2 得到学生模型和教师模型的feature map
#         feature_map_stu = super(DeformableDETRWithDistillation, self).extract_feat(batch_inputs)
#         feature_map_teacher = self.teacher_model.extract_feat(batch_inputs)
        
#         # 3 得到ground truth 的label
#         data_samples = kwargs["data_samples"]
        
#         # 初始化空列表，用来存储所有的 labels 和 bboxes
#         all_labels = []
#         all_bboxes = []

#         # 遍历每个数据样本
#         for sample in data_samples:
#             # 提取 gt_instances
#             gt_instances = sample.gt_instances

#             # 提取当前样本的 labels 和 bboxes
#             labels = gt_instances.labels
#             bboxes = gt_instances.bboxes
            
#             # 这里加入图片的原始尺寸 对每个bbox都加入原始尺寸(遍历添加)
#             ori_shape = sample.ori_shape
            
#             # 先遍历bboxes的每个bbox, 得到bbox之后，看怎么加入原始尺寸 original shape
#             x = torch.tensor(ori_shape[1], device='cuda:0')  
#             y = torch.tensor(ori_shape[0], device='cuda:0')  

#             # 将 x 和 y 扩展为与 bboxes 相同的行数
#             x_expanded = x.expand(bboxes.size(0), 1)
#             y_expanded = y.expand(bboxes.size(0), 1)
            
#             bboxes = torch.cat([bboxes[:, :4], x_expanded, y_expanded], dim=1)
                 
#             # 将当前样本的 labels 和 bboxes 添加到对应的列表中
#             all_labels.append(labels)
#             all_bboxes.append(bboxes)

#         # 将所有的 labels 和 bboxes 合并成一个大的 tensor
#         all_labels = torch.cat(all_labels, dim=0)  # 合并所有 labels
#         all_bboxes = torch.cat(all_bboxes, dim=0)  # 合并所有 bboxes
        
#         #返回的features_stu的形状是(m,通道数*num_scales)
#         features_stu, empty_flag_stu = self.extract_bbox_features(feature_map_stu,all_bboxes)
        
#         # 还差返回老师的特征feature,变量名字需要重新修改一下
#         features_teacher, empty_flag_teacher = self.extract_bbox_features(feature_map_teacher,all_bboxes)
        
#         # 如果没有提取到特征，则损失为0
#         if empty_flag_stu or empty_flag_teacher:
#             zero_tensor = torch.tensor(0.0)
#             student_output['prototype_loss'] = zero_tensor
#             return student_output
        
#         # 得到原型损失
#         pro_loss = self.prototype_loss(features_stu,features_teacher,all_labels,6)
#         student_output['prototype_loss'] = pro_loss
    
        return student_output

    def forward_test(self, *args, **kwargs):
        return super(DeformableDETRWithDistillation, self).forward(*args, **kwargs)
    
    
    def extract_bbox_features(self, feature_maps, bboxes):
        """
        从多尺度特征图中提取目标框的特征。

        :param feature_maps: 特征图列表，每个特征图形状为 (N, C, H, W)
        :param bboxes: 目标框的坐标，形状为 (m, 4)，每个框 [x_min, y_min, x_max, y_max]

        :return: 提取的目标框特征，形状为 (m, C * num_scales)
        """
        # m 是目标框的数量
        m = bboxes.shape[0]
        
        # num_scales 为4， 多尺度特征图的数量
        num_scales = len(feature_maps)
        
        # 待返回的特征
        features = []
        for i in range(m):
            # 获取每个目标框的坐标  这里加2个变量 图片的宽高
            x_min, y_min, x_max, y_max,image_width,image_height = bboxes[i]

            bbox_features_per_scale = []

            # 对每个特征图进行处理 总共有4个
            for feature_map in feature_maps:
                _, _, H_f, W_f = feature_map.shape  # 当前特征图的高度和宽度

                # 将目标框坐标归一化到当前特征图的尺寸 [0, W_f] 和 [0, H_f]
                x_min_f = (x_min * W_f / image_width)
                y_min_f = (y_min * H_f / image_height)
                x_max_f = (x_max * W_f / image_width)
                y_max_f = (y_max * H_f / image_height)

                # 转换为Tensor类型后进行处理
                x_min_f = torch.tensor(x_min_f, dtype=torch.float16)
                y_min_f = torch.tensor(y_min_f, dtype=torch.float16)
                x_max_f = torch.tensor(x_max_f, dtype=torch.float16)
                y_max_f = torch.tensor(y_max_f, dtype=torch.float16)

                # 确保坐标在特征图的边界内
                x_min_f = torch.max(torch.tensor(0.0), x_min_f)
                y_min_f = torch.max(torch.tensor(0.0), y_min_f)
                x_max_f = torch.min(torch.tensor(W_f - 1.0), x_max_f)  # W_f - 1 是最大有效索引
                y_max_f = torch.min(torch.tensor(H_f - 1.0), y_max_f)  # H_f - 1 是最大有效索引

                # 使用 floor 或 ceil 来确保坐标有效
                x_min_f = torch.floor(x_min_f).long()  # 向下取整
                y_min_f = torch.floor(y_min_f).long()
                x_max_f = torch.ceil(x_max_f).long()  # 向上取整
                y_max_f = torch.ceil(y_max_f).long()

                # 确保坐标顺序正确（左上角坐标小于右下角坐标）
                x_min_f = torch.min(x_min_f, x_max_f)
                y_min_f = torch.min(y_min_f, y_max_f)
                x_max_f = torch.max(x_min_f, x_max_f)
                y_max_f = torch.max(y_min_f, y_max_f)
                
                if x_min_f >= x_max_f or y_min_f >= y_max_f:
                    
                    y_min_f = y_max_f - 1
                    x_min_f = x_max_f - 1

                # 从特征图中提取目标框区域的特征
                bbox_feature = feature_map[:, :, y_min_f:y_max_f, x_min_f:x_max_f]

                # 对区域内的特征取均值
                bbox_feature = bbox_feature.mean(dim=[2, 3])  # 对区域内的特征取均值
                bbox_features_per_scale.append(bbox_feature)
            
            # 将所有尺度的特征拼接成一个长的向量
            combined_features = torch.cat(bbox_features_per_scale, dim=-1)  # 形状变为 (N, C * num_scales)
            features.append(combined_features[0])
            
        empty_flag = False
            
        # 判断feature列表的长度，如果为0，代表没有特征，那么损失为0
        if len(features) == 0:
            empty_flag = True
            return features, empty_flag

        return torch.stack(features, dim=0), empty_flag  # (m, C * num_scales)
    
    
    # !!!!还差教师模型的特征传入，函数的形参需要改!!!!
    def prototype_loss(self, features_stu, features_teacher, gt_labels, num_classes):
        """
        原型损失函数，计算目标框特征与类别原型的距离。

        :param features_stu: 目标框特征，形状为 (m, feature_dim)，其中 m 是目标框数量，feature_dim 是每个目标框的特征维度
        :param features_teacher: 目标框特征，形状为 (m, feature_dim)，其中 m 是目标框数量，feature_dim 是每个目标框的特征维度
        :param gt_labels: 目标框的类别标签，形状为 (m,) 每个目标框的类别
        :param num_classes: 类别数量

        :return: 原型损失值
        """
        # 计算每个类别的学生原型（类别特征的均值） ，!!!!目前还差教师模型的原型，变量名字也需要改!!!!
        prototypes_stu = []
        prototypes_teacher = []
        
        for c in range(num_classes):
            # 找到属于该类别的所有目标框(教师模型和学生模型)
            class_features_stu = features_stu[gt_labels == c]
            class_features_teacher = features_teacher[gt_labels == c]
            
            # 计算学生模型的原型
            if class_features_stu.size(0) > 0:
                class_prototype_stu = class_features_stu.mean(dim=0)  # 计算类别的原型
            else:
                class_prototype_stu = torch.zeros(features_stu.size(1)).to(features_stu.device)  # 如果没有该类别的目标框，设为零向量
            
            # 计算教师模型的原型
            if class_features_teacher.size(0) > 0:
                class_prototype_teacher = class_features_teacher.mean(dim=0)  # 计算类别的原型
            else:
                class_prototype_teacher = torch.zeros(features_teacher.size(1)).to(features_teacher.device)  
                
            # 检查计算出的原型是否包含nan
            if torch.isnan(class_prototype_stu).any():
                print(f"Warning: NaN found in class {c} prototype!")
                
            prototypes_stu.append(class_prototype_stu)
            prototypes_teacher.append(class_prototype_teacher)
        
        # 得到学生原型
        prototypes_stu = torch.stack(prototypes_stu, dim=0)  # 形状为 (num_classes, feature_dim)
        #得到教师模型
        prototypes_teacher = torch.stack(prototypes_teacher, dim=0)  # 形状为 (num_classes, feature_dim)
        
        # 计算每个目标框特征与其对应类别的原型之间的余弦距离
        losses = []
        for i in range(features_stu.size(0)):
            label = gt_labels[i]
            
            prototype_stu = prototypes_stu[label]  # 获取当前目标框的学生模型的类别原型
            feature_stu = features_stu[i]  # 获取当前目标框学生模型的特征
            
            prototype_teacher = prototypes_teacher[label]  # 获取当前目标框的教师模型的类别原型
            feature_teacher = features_teacher[i]  # 获取当前目标框教师模型的特征

            # 归一化特征和原型，避免零向量的问题
            feature_norm_stu = torch.norm(feature_stu)
            prototype_norm_stu = torch.norm(prototype_stu)
            
            feature_norm_teacher = torch.norm(feature_teacher)
            prototype_norm_teacher = torch.norm(prototype_teacher)

            # 如果特征向量和原型向量都非零，计算余弦相似度， !!!!还需计算教师模型的余弦相似度!!!!
            if feature_norm_stu > 1e-6 and prototype_norm_stu > 1e-6 and feature_norm_teacher > 1e-6 and prototype_norm_teacher > 1e-6 :
                cosine_similarity_stu = F.cosine_similarity(feature_stu.unsqueeze(0), prototype_stu.unsqueeze(0), dim=-1)
                cosine_similarity_teacher = F.cosine_similarity(feature_teacher.unsqueeze(0), prototype_teacher.unsqueeze(0), dim=-1)
                
                pro_loss = (cosine_similarity_teacher - cosine_similarity_stu) ** 2
                losses.append(pro_loss)  
                
            else:
                losses.append(torch.tensor(0.0).to(features_stu.device))  # 如果为零向量，损失为0
        
        # 返回所有目标框的原型损失平均值
        return torch.mean(torch.stack(losses))
    
