# 将所有要显示的图片都集中到一个窗口显示

from matplotlib import pyplot as plt
import tkinter

# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'SimHei'


# 传入图像列表(各元素名字不能相同)，将各个图像显示出来;"covert_rgb"参数表示是否要转换成rgb格式
def plt_pic(img_dict, covert_rgb=True):
    assert isinstance(img_dict, dict), 'plt_pic函数第一个传入参数不是字典类型'
    assert isinstance(covert_rgb, bool), 'plt_pic函数第二个传入参数不是布尔类型'
    _length = len(img_dict)
    # 判断图片排版类型
    if _length <= 3:
        layout = (1, _length)
    elif _length == 4:
        layout = (2, 2)
    elif 5 <= _length <= 6:
        layout = (2, 3)
    elif 7 <= _length <= 9:
        layout = (3, 3)
    else:
        raise ValueError("plt_pic函数传入参数的长度必须≤9")

    screen = tkinter.Tk()
    width = screen.winfo_screenwidth()              # 获取当前屏幕的宽
    height = screen.winfo_screenheight()            # 获取当前屏幕的高
    screen.destroy()
    w = (width-200) / 80
    h = (height-200) / 80

    _names = list(img_dict.keys())
    plt.close('all')               # 关闭所有画布figure
    plt.figure('图像对比', figsize=(w, h), dpi=80)  # 如果不传入参数默认画板1,"figsize"代表画布尺寸，默认(6.4,4.8)
    mngr = plt.get_current_fig_manager()     # 获取当前figure manager
    mngr.window.wm_geometry("+50+50")      # 调整窗口在屏幕上弹出的位置
    for _i in range(_length):
        _name = _names[_i]
        plt.subplot(layout[0], layout[1], _i+1)
        if covert_rgb and img_dict[_name].ndim == 3:
            plt.imshow(img_dict[_name][:, :, (2, 1, 0)])
        elif covert_rgb and img_dict[_name].ndim == 2:
            plt.imshow(img_dict[_name], cmap="gray")
        else:
            plt.imshow(img_dict[_name])
        plt.title(_name, fontsize=20)
        plt.axis('off')
    plt.show()
