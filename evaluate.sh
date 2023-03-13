# "${1}"=output model dir name_prefix: base_, base_EOS, large_, large_EOS

year='15'
if [[ "$1" == *"2016"* ]]; then
  year='16'
fi

echo "creating combined prediction file"

# python merge_pred_files.py semeval-20"${year}" TASD sentence "${1}"

echo "Evaluating the TASD, ASD, TSD tasks"

python evaluation_for_TSD_ASD_TASD.py --output_dir results/TASD"${1}" --tag_schema TO --num_epochs 0 > results/TASD"${1}"/TASD1.txt
python evaluation_for_TSD_ASD_TASD.py --output_dir results/TASD"${1}" --tag_schema TO --num_epochs 1 > results/TASD"${1}"/TASD2.txt
python evaluation_for_TSD_ASD_TASD.py --output_dir results/TASD"${1}" --tag_schema TO --num_epochs 2 > results/TASD"${1}"/TASD3.txt
python evaluation_for_TSD_ASD_TASD.py --output_dir results/TASD"${1}" --tag_schema TO --num_epochs 4 > results/TASD"${1}"/TASD4.txt

cd evaluation_for_AD_TD_TAD

echo "Creating the XML pred files"

python change_pre_to_xml.py --gold_path ../data/semeval-20"${year}"/test_TAS.tsv --pre_path ../results/TASD"${1}"/converted_predictions0.txt --gold_xml_file ABSA"${year}"_Restaurants_Test.xml --pre_xml_file pred_file_20"${year}"_T5_"${1}"0.xml --tag_schema TO
python change_pre_to_xml.py --gold_path ../data/semeval-20"${year}"/test_TAS.tsv --pre_path ../results/TASD"${1}"/converted_predictions1.txt --gold_xml_file ABSA"${year}"_Restaurants_Test.xml --pre_xml_file pred_file_20"${year}"_T5_"${1}"1.xml --tag_schema TO
python change_pre_to_xml.py --gold_path ../data/semeval-20"${year}"/test_TAS.tsv --pre_path ../results/TASD"${1}"/converted_predictions2.txt --gold_xml_file ABSA"${year}"_Restaurants_Test.xml --pre_xml_file pred_file_20"${year}"_T5_"${1}"2.xml --tag_schema TO
python change_pre_to_xml.py --gold_path ../data/semeval-20"${year}"/test_TAS.tsv --pre_path ../results/TASD"${1}"/converted_predictions4.txt --gold_xml_file ABSA"${year}"_Restaurants_Test.xml --pre_xml_file pred_file_20"${year}"_T5_"${1}"4.xml --tag_schema TO

echo "******************************************** 1st Sentence Results ***********************************************"

pred_file_name="${1}"
echo "%%%%%%%%%%%%%%%%%%%%%%% ${pred_file_name} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
if [[ "${1}" == *"best_model"* ]]; then
  echo "------------ It's there. ----------------------"
  IFS='/'
  read -a strarr <<< "${1}"
  pred_file_name="${strarr[0]}_${strarr[1]}"
  echo "------------------------ ${pred_file_name} --------------------------------------------------"
fi

cat ../results/TASD"${1}"/TASD1.txt
java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}0.xml ./ABSA"${year}"_Restaurants_Test.xml 1 0 > ../results/TASD"${1}"/AD1.txt
echo "Aspect Detection:\n"
tail -7 ../results/TASD"${1}"/AD1.txt

java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}0.xml ./ABSA"${year}"_Restaurants_Test.xml 2 0 > ../results/TASD"${1}"/TD1.txt
echo "Target Detection:\n"
tail -7 ../results/TASD"${1}"/TD1.txt

java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}0.xml ./ABSA"${year}"_Restaurants_Test.xml 3 0 > ../results/TASD"${1}"/TAD1.txt
echo "Target Aspect Detection:\n"
tail -7 ../results/TASD"${1}"/TAD1.txt

echo "******************************************** 2nd Sentence Results ***********************************************"

cat ../results/TASD"${1}"/TASD2.txt
java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}1.xml ./ABSA"${year}"_Restaurants_Test.xml 1 0 > ../results/TASD"${1}"/AD2.txt
echo "Aspect Detection:\n"
tail -7 ../results/TASD"${1}"/AD2.txt

java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}1.xml ./ABSA"${year}"_Restaurants_Test.xml 2 0 > ../results/TASD"${1}"/TD2.txt
echo "Target Detection:\n"
tail -7 ../results/TASD"${1}"/TD2.txt

java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}1.xml ./ABSA"${year}"_Restaurants_Test.xml 3 0 > ../results/TASD"${1}"/TAD2.txt
echo "Target Aspect Detection:\n"
tail -7 ../results/TASD"${1}"/TAD2.txt

echo "******************************************** 3rd Sentence Results ***********************************************"

cat ../results/TASD"${1}"/TASD3.txt
java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}2.xml ./ABSA"${year}"_Restaurants_Test.xml 1 0 > ../results/TASD"${1}"/AD3.txt
echo "Aspect Detection:\n"
tail -7 ../results/TASD"${1}"/AD3.txt

java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}2.xml ./ABSA"${year}"_Restaurants_Test.xml 2 0 > ../results/TASD"${1}"/TD3.txt
echo "Target Detection:\n"
tail -7 ../results/TASD"${1}"/TD3.txt

java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}2.xml ./ABSA"${year}"_Restaurants_Test.xml 3 0 > ../results/TASD"${1}"/TAD3.txt
echo "Target Aspect Detection:\n"
tail -7 ../results/TASD"${1}"/TAD3.txt

echo "******************************************** Combined Sentences Results *****************************************"

cat ../results/TASD"${1}"/TASD4.txt
java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}4.xml ./ABSA"${year}"_Restaurants_Test.xml 1 0 > ../results/TASD"${1}"/AD4.txt
echo "Aspect Detection:\n"
tail -7 ../results/TASD"${1}"/AD4.txt

java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}4.xml ./ABSA"${year}"_Restaurants_Test.xml 2 0 > ../results/TASD"${1}"/TD4.txt
echo "Target Detection:\n"
tail -7 ../results/TASD"${1}"/TD4.txt

java -cp ./A.jar absa15.Do Eval ./pred_file_20"${year}"_T5_${pred_file_name}4.xml ./ABSA"${year}"_Restaurants_Test.xml 3 0 > ../results/TASD"${1}"/TAD4.txt
echo "Target Aspect Detection:\n"
tail -7 ../results/TASD"${1}"/TAD4.txt


cd ..