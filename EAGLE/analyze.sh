# for every file in mt_bench folder, we run the eagle.evaluation.accept_length python script
# then, we save the output to log.txt file along with the file name

for file in mt_bench/*; do
    echo $file >> log.txt
    python eagle/evaluation/accept_length.py --jsonl_file $file >> log.txt
done