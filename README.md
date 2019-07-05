# Mtech_Project
**Command to train model:**
python ProGCN/train_latest.py 256 256 0.1 0.9 0.0001 ProGCN/models/ ProGCN/results/
**Format:**
python <filename> <lstm_dim> <gcn_dim> <dropout> <recurrent_dropout> <learning_rate> <model_save_location> <result_save_location>

**Command to evaluate test file -**
python ProGCN/evalQA.py propara_naacl_2018/tests/fixtures/eval/para_id.test.txt propara_naacl_2018/tests/fixtures/eval/gold_labels.test.tsv ProGCN/data/propara-results-test_latest.txt

**Sample Output -**

    |	Total	| TP	| FP	| TN	| FN	| Accuracy	| Precision	| Recall	| F1 |
    | ----- | --- | ----| --- | ----| --------- | --------- | ------  | ---|
Q1  |1245	  |0    |	0	  | 577	| 668 |	46.35	    | 0.00      |0.00     |0.00|
Q2	677	0	0	0	677	0.00	0.00	0.00	0.00
Q3	529	148	194	0	187	27.98	43.27	44.18	43.72
Q4	1245	220	200	358	467	46.43	52.38	32.02	39.75
Q5	709	15	206	0	488	2.12	6.79	2.98	4.14
Q6	546	149	193	0	204	27.29	43.57	42.21	42.88
Q7	1245	214	271	499	261	57.27	44.12	45.05	44.58
Q8	3422	79	197	2484	662	15.54	28.62	10.66	15.54
Q9	549	99	275	0	175	18.03	26.47	36.13	30.56
Q10	691	143	363	0	185	20.69	28.26	43.60	34.29



#Category	Accuracy Score
#=========	=====
#Cat-1		50.01
#Cat-2		5.88
#Cat-3		23.5
#macro-avg	26.47
#micro-avg	27.78
