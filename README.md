# task26100285

## glb_measure — GLB 静态几何计量库

供资产导入流程直接调用的纯 Python 库（仅标准库，无第三方依赖）。
输入本地 GLB 文件，返回默认（或第一个）scene 实际参与绘制的
三角面总数，以及以英寸表示的世界坐标轴对齐包围盒；不保存任何
派生数据。

### 用法

```python
from glb_measure import measure_glb

result = measure_glb("/path/to/asset.glb")
print(result.triangle_count)   # 精确三角面数
if result.bbox is not None:
    print(result.bbox.min)     # (x, y, z) 英寸，世界坐标
    print(result.bbox.max)
else:
    print("场景中没有三角面")   # 面数为 0 时包围盒为 None
```

### 计量规则

- 遵循 glTF 2.0 的 GLB 语义；二进制资源全部来自内嵌 BIN 块。
- 指定默认 scene 时只计量该 scene，否则取第一个 scene；只计量其
  根节点及后代引用的网格。
- 同一 mesh 被多个节点引用时按实例分别计量；每个实例应用节点的
  `matrix` 或 TRS，并叠加全部祖先变换（旋转、负缩放、非均匀缩放
  均正确处理）。
- 只计量 `TRIANGLES` / `TRIANGLE_STRIP` / `TRIANGLE_FAN`（mode 省略
  按 `TRIANGLES`）；退化三角面仍计一面，其顶点参与范围计算。
- 支持 bufferView/accessor 偏移、交错 `byteStride`，以及 POSITION
  与索引上的稀疏 accessor（含无基础 bufferView 时按零补齐）。
- 包围盒的每个边界都由实际参与三角面的世界坐标顶点取得；未被
  索引引用的顶点不会扩大结果。
- 坐标单位按米解释，输出换算为英寸（1 英寸 = 0.0254 米）。

### 非功能特性

- BIN 块通过 `mmap` 按需读取：500MB 单文件、大型内嵌图片不会
  使计量内存随其体积增长（单次计量峰值内存增量远低于 128MiB）。
- 全程 float64 计算；世界坐标绝对值不超过 10000 米时，边界坐标
  绝对误差不超过 0.0001 英寸。

### 测试

```bash
python3 -m pytest tests/ -q
```

覆盖：旋转网格忽略未引用极值顶点、同一 mesh 多实例分别计数、
稀疏覆盖（有/无基础 bufferView、稀疏索引）、交错布局、条带/扇形
拓扑、退化三角面、空结果、scene 选择，以及接近 500MB 文件的
内存验证（结果不受无关图片与未选中 scene 影响）。
