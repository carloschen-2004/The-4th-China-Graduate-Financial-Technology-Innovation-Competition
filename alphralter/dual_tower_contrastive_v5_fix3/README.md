"dual\_tower\_contrastive\_v5\_fix3" 目录下打开终端



### 指令



**训练：**
python train\_per\_category.py  --data\_dir 'raw\_data' --save\_dir 'outputs'  --epochs 2 --batch\_size 64 --lr 3e-4 --temperature 1.0 --use\_hard\_negative True --margin 0.05 --neg\_weight 0.001

**预测：**
python recommend\_user\_profiles.py --data\_dir 'data'  --model\_dir 'outputs'  --out\_csv 'outputs/recommended\_user\_profiles.csv'  --k  60

**检验预测准确率：**
python accuracy\_test.py



### 文件存储说明



1. 原始数据 **raw\_data**  
   客户表(cust\_dataset), 事件表(event\_dataset), 产品表(productLabels\_multiSpreadsheets)
2. 训练脚本  
   **utils:** data\_cleaner\_v5, feature\_align\_v5 用于清洗数据
   **train\_per\_category:** 用于对每一个产品类别进行训练
   **recommend\_user\_profiles:** 用于预测产品可行的用户list
3. 训练-预测生成数据  
   **cleaned:** 事件、客户数据清洗结果
   **aligned:** 产品表清洗合并结果
   **merged:** 事件、客户、产品合并结果
   **outputs:** 模型(model\_A/C/D/N/P)、训练日志(train\_log)及预测结果(recommended\_user\_profiles, recommended\_user\_profiles\_acc, acc为包含准确率的预测结果)
