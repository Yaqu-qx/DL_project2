## project2 记录
### 项目结构
- dataset-basic: 低级任务数据
- dataset-advance:高级任务数据
- advance-output: 高级任务实验结果
- basic-ouput: 基础任务实验结果
- mast3r: mast3r项目目录
- SuperGluePretrainedNetwork：SuperGlue项目目录
- vggt: vggt项目目录
- colmap_to_superglue.py: 用于basic task中位姿转化脚本，可以输入多个图像对，输出一个符合superglue的测试集，即一个$.txt$文件。需要在命令中指定colmap原始位姿数据路径（--colmap_path）、输出文件路径（--output_file）。可选择在命令行中指定要匹配的图片对数（--num_pairs），已经对应匹配对（--set-pairs）。默认匹配10对，按顺序两两匹配。
e.g.
```bash
python colmap_to_superglue.py --colmap_path data/output/sparse/0 --output_file my_superglue_pairs.txt --num_pairs 4 --set-pairs 01 02 03 04
```
- superglue_format.py:用于advance task中位姿格式转化成符合superglue输入格式。
- project2/dataset-advance/bdaibdai___MatrixCity/small_city/aerial/test/block_1_test: 高级任务1匹配图像对
- project2/dataset-advance/bdaibdai___MatrixCity/aerial_street_fusion/pairs.py: 高级任务2匹配图像对 
- project2/dataset-advance/road/images/aerial_h: 高级任务3匹配图像对
- project2/SuperGluePretrainedNetwork/match_pairs.py: SuperGlue图像匹配文件
- project2/mast3r/pair_match.py：mast3r图像匹配文件
- project2/vggt/pair_match.py：vggt图像匹配文件

#### run
##### SuperGlue
```bash
cd SuperGluePretrainedNetwork
./match_pairs.py --input_pairs ../my_superglue_pairs.txt --input_dir ../data/input --output_dir ../data/superglue_output --superglue outdoor --viz --eval
```
##### Mast3r
```bash
cd mast3r --weight project2/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
python pair_match.py
```
##### vggt
```bash
cd vggt
python pair_match.py
```