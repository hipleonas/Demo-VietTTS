import os
import shutil


if  __name__ == "__main__":


    src_wav_file = "/home/hiepquoc/voice_data/vivos/train/waves"

    src_prompt_file = "/home/hiepquoc/voice_data/vivos/train/prompts.txt"
    
    dst_dir = "/home/hiepquoc/voice_data/prepared_vivos/data"

    os.makedirs(dst_dir, exist_ok= True)

    #Loadding the prompt file
    print('Load prompt file from')
    prompt_hash = {}

    with open(src_prompt_file, "r", encoding="utf-8") as file:

        for line in file:
            # if line.strip() == "":
            #     continue  #  Skip empty lines      
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                file_id, text = parts
                prompt_hash[file_id] = text.lower()        
    # #Copy and process
    # print(prompt_hash)
    count = 1

    for file_id, text in prompt_hash.items():
        file_id = file_id.split("_")  # Remove any file extension if present
        speaker_id, wav_id = file_id[0], file_id[1] 
        wav_path = os.path.join(src_wav_file, f"{speaker_id}",f"{speaker_id}_{wav_id}.wav")
        print(wav_path)        
        if not os.path.exists(wav_path):
            continue
        # print('yes')
        target_id = f"{count:03}"
        shutil.copy(wav_path, os.path.join(dst_dir, f"{target_id}.wav"))
        with open(os.path.join(dst_dir, f"{target_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        count += 1
    print(f"Đã xử lý xong {count-1} cặp wav + txt vào thư mục `data/`.")
