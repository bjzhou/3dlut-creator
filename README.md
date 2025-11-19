# 3D LUT Creator

从图片对生成3D查找表(LUT)的工具，支持多种格式输出。

## 功能特点

- 🎨 从图片对自动提取RGB颜色映射关系
- 📊 生成高质量64×64×64 3D LUT (可自定义尺寸)
- 🔧 支持多种LUT格式导出 (.cube, .3dl, .dat, .npy)
- 📈 智能线性插值算法，确保平滑过渡
- 🖼️ 可选生成LUT预览图像
- ⚡ 支持批量处理多对图片
- 🚀 **GPU加速支持** - 使用PyTorch和MPS/CUDA实现4-5倍性能提升

## 工作原理

1. **基准映射**: 以 `photoa` 文件夹的图片RGB值作为输入
2. **颜色映射**: 将 `photob` 文件夹对应图片的RGB值作为输出
3. **智能插值**: 使用增强的K近邻算法进行三线性插值
4. **3D网格**: 构建64×64×64的3D网格存储颜色映射关系

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python main.py --photoa photoa_folder --photob photob_folder --output my_lut
```

### 完整参数

```bash
python main.py \
    --photoa ./photoa \
    --photob ./photob \
    --output ./output/color_grade \
    --size 64 \
    --formats cube,3dl \
    --title "My Color Grade" \
    --device auto
```

### 参数说明

- `--photoa`: 基准RGB图片文件夹路径 (输入颜色) **必需**
- `--photob`: 映射RGB图片文件夹路径 (输出颜色) **必需**
- `--output`: 输出LUT文件的基础名称 (不含扩展名) **必需**
- `--size`: LUT网格大小 (默认: 64)
- `--formats`: 导出格式: cube,3dl,dat,npy,all (默认: all)
- `--title`: LUT文件标题 (默认: "3D LUT from Image Pairs")
- `--device`: 计算设备: auto, mps, cuda, cpu （默认: auto）

## 支持的图片格式

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## 支持的LUT格式

| 格式 | 扩展名 | 说明 | 兼容软件 |
|------|--------|------|----------|
| Adobe Cube | .cube | 最常用的LUT格式 | Premiere Pro, After Effects, DaVinci Resolve, Photoshop |
| 3DL | .3dl | 传统LUT格式 | SpeedGrade, Nuke |
| DaVinci DAT | .dat | DaVinci Resolve格式 | DaVinci Resolve |
| NumPy | .npy | Python二进制格式 | 用于程序间数据交换 |

## 推荐配置

### 图片尺寸建议

为了获得最佳的3D LUT质量，建议遵循以下图片尺寸配置：

| 尺寸 | 像素数 | 处理时间 | 内存使用 | LUT质量 | 推荐场景 |
|------|--------|----------|----------|---------|----------|
| 400×300 | 12万 | 快 | 低 | 基础 | 快速测试 |
| 800×600 | 48万 | 中等 | 中等 | 良好 | 一般使用 |
| 1280×720 | 92万 | 中等 | 中等 | 很好 | ✅ **推荐使用** |
| 1920×1080 | 207万 | 慢 | 高 | 优秀 | 专业质量 |
| 2560×1440 | 368万 | 很慢 | 很高 | 极佳 | 特殊需求 |

### 📐 图片规格要求

- **尺寸**: 1280×720 (推荐) 到 1920×1080
- **格式**: PNG (无损) 或高质量JPEG
- **色深**: 8-bit per channel
- **色彩空间**: sRGB

### 🎨 图片内容建议

为了获得最佳的LUT效果，建议使用以下类型的图片：

1. **颜色多样性** (3-5张)
   - 包含丰富的色彩变化
   - 覆盖色相环的各个部分
   - 避免大面积纯色区域

2. **代表性场景** (2-3张)
   - 人物肖像 (肤色色调)
   - 自然风景 (绿色和蓝色)
   - 建筑或街景 (中性色调)

3. **特殊光照** (1-2张)
   - 高对比度场景 (明暗对比)
   - 黄金时刻的暖色调
   - 夜景的冷色调

4. **渐变测试** (1张)
   - 包含平滑渐变
   - 测试插值算法的平滑性

### ⚡ 性能优化建议

- **减少内存使用**: 使用较小的LUT size (32-64)
- **加快处理**: 减少10张图片到3-5张高质量图片
- **提高精度**: 使用更高分辨率的图片 (1920×1080)

## 使用示例

### 1. 准备图片对

确保 `photoa` 和 `photob` 文件夹中有相同文件名的图片对：

```
photoa/
├── scene1.jpg      # 1920×1080 风景照片
├── portrait.png    # 1280×720 人物肖像
├── gradient.tif    # 800×600 渐变测试图
└── high_contrast.jpg # 1280×720 高对比度

photob/
├── scene1.jpg      # 调色后的风景照片
├── portrait.png    # 调色后的人物肖像
├── gradient.tif    # 调色后的渐变
└── high_contrast.jpg # 调色后的高对比度
```

### 2. 生成LUT

```bash
python main.py --photoa ./photoa --photob ./photob --output ./output/film_lut
```

### 3. 使用生成的LUT

将生成的 `.cube` 文件导入到支持LUT的软件中：

- **Adobe Premiere Pro**: 效果面板 → Lumetri颜色 → 输入LUT
- **DaVinci Resolve**: 色彩页面 → LUTs → 添加3D LUT
- **Adobe Photoshop**: 图层 → 新建调整图层 → 颜色查找 → 3DLUT文件

## 技术细节

### 插值算法

本工具使用增强的K近邻插值算法：

- **K近邻搜索**: 找到最近的8个已知颜色点
- **高斯权重**: 使用高斯函数计算距离权重
- **加权平均**: 对多个映射值进行平滑插值

### 内存使用

- LUT大小 32: ~1MB内存
- LUT大小 64: ~8MB内存
- LUT大小 128: ~64MB内存

### 性能优化

- 支持并行处理多张图片
- 使用NumPy向量化计算
- 智能缓存颜色映射关系

## 故障排除

### 常见问题

1. **没有找到配对图片**
   - 检查文件名是否完全匹配
   - 确认文件扩展名正确

2. **内存不足**
   - 减小LUT大小 (如使用32而不是64)
   - 处理较少的图片对

3. **输出颜色不准确**
   - 确保输入图片为sRGB色彩空间
   - 检查图片是否有过度的后期处理

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0
- 首次发布
- 支持图片对到3D LUT的转换
- 支持多种LUT格式导出
- 实现智能插值算法