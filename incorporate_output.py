def create_csv_file(input_filename, output_filename):
    with open(input_filename, "r") as input_file, open(output_filename, "w") as output_file:
        # 写入 CSV 文件的标题行
        output_file.write("PhraseId,Sentiment\n")

        # 逐行读取 input_filename 中的数字，并将其写入 CSV 文件
        phrase_id = 156061
        for line in input_file:
            for ch in line:

                output_file.write(f"{phrase_id},{ch}\n")

                # PhraseId 按照递增顺序自增
                phrase_id += 1



if __name__=='__main__':

    # 在调用函数时传入 input.txt 的文件路径和输出 CSV 文件的路径
    create_csv_file("output_100_2.txt", "movie_Submission_2.csv")




