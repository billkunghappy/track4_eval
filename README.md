# track4_eval
* 可以 evaluate response(BLEU, retrieval score) 以及 Belief state score
## 準備
* 把 devtest predict 的檔案放入對應的資料夾，比如furniture 類型的檔案
  * furniture_devtest_dials_predicted.txt 放到 track4_eval/results/furniture
  * 這邊為了檔案方便區分，建議predict檔案都有個postfix，比如有用tid的叫做 furniture_devtest_dials_predicted_tid.txt
  * 若是 predict 檔案不存在<EOB>，需要加在對應的位置方便eval script eval，跑 python3 scripts/add_eob.py --file_name file_name
    * add_eob.py 預設是把 <EOB> 加在最後一個 "System : " 的後面，可以依自己需求修改
### retrieval candidates
* 跑 ./get_retrieve_scores_BLEU.sh ${type} ${postfix}
  * ${type} 對應到數字 [0,1,2,3]，分別代表 [furniture_to, furniture_to, fashion_to, fasion]
  * ${postfix} 是加在predict檔案的.txt前面的字串
  * 比如要跑 furniture 類型的 furniture_devtest_dials_predicted_tid.txt 檔案的話
    * 執行 ./get_retrieve_scores_BLEU.sh 1 "_tid"
  * 結束後會在 results/${type}/ 資料夾生成candidates score

## 執行evaluate
* ./run_evaluate_gpt2.sh ${type} ${postfix}
  * 可以在 run_evaluate_gpt2.sh 加三個參數：
    * --skip_belief   --> 不eval belief 
    * --skip_bleu     --> 不eval bleu
    * --skip_retr     --> 不eval retrieval scores 
