#!/bin/bash
set -e

task_file="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/task_2src_list.txt"

# 输出文件
final_output="found_list.txt"
missing_output="missing_list.txt"

# 初始化
> "$final_output"
> "$missing_output"

# 写表头
echo -e "demography\tsample\tcutoff\tsrc\tprecision\trecall" > "$final_output"

# 遍历
while read -r DEMOG NREF NTGT SEED; do
    # 去掉可能多余的空格
    DEMOG=$(echo "$DEMOG" | tr -d '\r')
    NREF=$(echo "$NREF" | tr -d '\r')
    NTGT=$(echo "$NTGT" | tr -d '\r')
    SEED=$(echo "$SEED" | tr -d '\r')

    acc_file="/home/linhuanyu/share1/20_AS3/results/inference/SkovHMM/${DEMOG}/nref_${NREF}/ntgt_${NTGT}/${SEED}/accuracy.txt"

    echo "🔎 Checking: $acc_file"  # 打印调试用

    if [[ -s "$acc_file" ]]; then
        echo "✅ Found: $acc_file"
        tail -n +1 "$acc_file" >> "$final_output"
    else
        echo "❌ Missing: $DEMOG $NREF $NTGT $SEED"
        echo "$DEMOG $NREF $NTGT $SEED" >> "$missing_output"
    fi
done < "$task_file"

echo "✅ Merge done: found_list.txt"
echo "❗ Missing list saved to: missing_list.txt"
