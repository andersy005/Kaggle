def create_submission(predictions, test_id, info):
    sub_file = os.path.join('submission_' + info + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    subm = open(sub_file, "w")
    mask = find_best_mask()
    encode = rle_encode(mask)
    subm.write("img,pixels\n")
    for i in range(len(test_id)):
        subm.write(str(test_id[i]) + ',')
        if predictions[i][1] > 0.5:
            subm.write(encode)
        subm.write('\n')
    subm.close()
