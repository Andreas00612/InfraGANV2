import matplotlib.pyplot as plt
import os

def find_different_indices(lst):
    return [0] + [i + 1 for i in range(len(lst) - 1) if lst[i] != lst[i + 1]]


def get_split_index(all_data_path):
    dir_name_list = []
    for data_path in all_data_path:
        path_names = data_path.split('/')[-1].split('\\')
        dir_name = path_names[1] + '_' + path_names[2]
        dir_name_list.append(dir_name)

    result = find_different_indices(dir_name_list)
    return result, dir_name_list


def draq_and_analysis(data, all_data_path, save_dir=None,mode='ssim'):
    split_indexs, dir_name_list = get_split_index(all_data_path)
    plt.figure()

    plt.scatter(range(len(data)), data, s=2)

    # 繪製垂直分割线
    for idx, split_idx in enumerate(split_indexs):
        _list = data[split_indexs[idx]:split_indexs[idx + 1]] if idx != len(split_indexs) - 1 else data[
                                                                                                   split_indexs[idx]:]
        average = sum(_list) / len(_list)

        draw_point = (split_indexs[idx] + split_indexs[idx + 1]) // 2 if idx != len(split_indexs) - 1 else len(data) - 1
        plt.axvline(x=split_idx, color='red', linestyle='--')

        plt.annotate(dir_name_list[split_idx], xy=(draw_point-5, min(data)), xytext=(0, -40), textcoords='offset points',
                     ha='center', va='top', rotation='vertical')

        plt.annotate(f'{average:.3f}', xy=(draw_point + 10, min(data)), xytext=(0, -40), textcoords='offset points',
                     ha='center', va='top', rotation='vertical')

    plt.xticks([])
    plt.ylabel(mode)
    fig = plt.gcf()
    fig.set_size_inches(25, 10)
    text_name = os.path.join(save_dir, mode + '_result.png')
    plt.savefig(text_name,dpi=400)



if __name__ == '__main__':
    data = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, ]
    draq_and_analysis(data)
    pass
