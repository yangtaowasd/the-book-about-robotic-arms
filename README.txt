机械臂控制与仿真详细教材——配套源文件与数值算例
整合与扩展版，2026-11-23

一、读什么

教材正文以单关节、竖直平面2R和七轴相机任务为主线，含23章正文、
4个附录、57幅图、114个框内算例、83道附逐步解答的章末练习。
PDF是完整阅读版本，不需要执行程序即可学习。
本压缩包包含可编辑正文、图的生成程序、数值校核以及合成实验数据。

两份用户提供的原稿在文中记为A和B，每章末列出对应PDF物理页码。
原稿本身未重复打包；正文为重新组织、推导和扩展后的整合版本。

二、目录和入口

textbook.tex                 XeLaTeX入口
frontmatter.tex              封面、阅读说明、符号与模型
chapters/                    23章正文
appendices.tex               附录A-C：来源、术语、公式条件与复算
reproducibility.tex          附录D：复现方法与最小代码
references.tex              原稿和补充的一手资料，含网页链接
figures/                    57组PDF矢量图与PNG预览
requirements.txt            本次数值计算使用的Python库版本

math_checks.py               第1-5章关键数值校核
dynamics_checks.py           第6-10章关键数值和结构校核
control_checks.py            第11-16章关键数值校核
minimal_2r_check.py           附录D的最小2R计算，可独立运行

math_figures.py              生成第1-5章图
dynamics_figures.py          生成第6-10章图
control_figures.py           生成第11-16章图
root_figures.py              生成后续章节的轨迹、示教、时序、MPC、云台图
integrated_experiment.py     第23章完整辨识与控制实验

*_results.json、*_numbers.json 计算结果与核对记录
root_numeric_checks.json     后续章节主要数字
integration_data.npz         合成训练和验证数据
integration_results.json     第23章参数、条件与最终性能指标
integration_metrics.tex      由实验程序生成的教材表格
artifact_manifest.json       文件规模及本版本信息

三、运行数值计算

在解压后的 robotics_textbook 目录运行。建议使用Python 3.12和独立虚拟环境。
requirements.txt记录了本次环境的NumPy、SciPy、SymPy、Matplotlib版本。
可使用：

    python -m pip install -r requirements.txt
    python minimal_2r_check.py
    python math_checks.py
    python dynamics_checks.py
    python control_checks.py
    python integrated_experiment.py

重新生成其他图：

    python math_figures.py
    python dynamics_figures.py
    python control_figures.py
    python root_figures.py

中文图使用Noto Sans CJK JP字体名（Noto CJK集合字体包含中文）；没有该字体时，
安装Noto CJK或修改各脚本的font.family为本机可用的中文字体。
中文图的PDF采用Type 3字形，以避免部分XeLaTeX环境处理CJK集合字体时丢字。
英文数学图使用DejaVu Sans。图的坐标与数值来自方程或明确给定的数据。

程序将重写自己对应的结果和图片，不连接机器人、不发送硬件命令。

四、编译正文

需要XeLaTeX（例如完整TeX Live环境），以及Noto Serif CJK SC、Noto Sans CJK SC、
DejaVu Sans Mono字体。常用宏包包括fontspec、amsmath、mathtools、booktabs、
longtable、tabularx、graphicx、tcolorbox、listings、hyperref、bookmark、
enumitem、titlesec、fancyhdr、geometry、setspace。

运行：

    xelatex -interaction=nonstopmode -halt-on-error textbook.tex
    xelatex -interaction=nonstopmode -halt-on-error textbook.tex

第一次生成目录和标签，第二次使交叉引用和目录页码一致。
如仍提示标签变化，再编译一次。已有的PDF图足以编译，无须先运行所有作图脚本。
不同字体、TeX发行版或浮点库可能造成很小的分页或末位数字差异。

五、第23章实验的准确含义

这是本地执行的教学仿真，使用显式2R方程与RK4。执行器为一阶模型，
其解析力矩在RK4各阶段求值；不是MuJoCo实验，不是真机测量。

名义杆长为0.5 m、质量各1 kg，质心为半长，质心惯量按均匀杆公式计算。
仿真对象质量为1.1和0.9 kg，带平滑库仑和黏性摩擦。
控制周期5 ms，物理步0.5 ms，延迟一个控制周期，执行器时间常数4 ms。
关节力矩限制为15与6 N m；延迟队列与执行器力矩从零初始化。

训练数据3000个时刻、6000条力矩方程，拟合两个质量与四个摩擦系数。
摩擦平滑尺度0.02 rad/s为已知设定。另一条轨迹与独立噪声用于验证。
位置、速度、加速度在辨识数据中取解析值；本实验不声称覆盖真机全部噪声链。

控制任务为三秒五次关节轨迹加一秒保持。0-4秒按200 Hz记录801点。
RMS是末端二维误差模长的均方根；末值是4秒时的误差。
饱和计数对应800个有效控制拍，每拍任一轴被裁剪计1。
比较方法、真实参数、拟合参数和最后数字以integration_results.json为准。

MuJoCo的MJCF示例在第21章用于解释模型结构，本次没有执行该XML。
没有NERO的实测参数、运行日志或驱动器内部数据，教材中的教学参数不能
当成该机械臂的实际标定结果。书中对此作了明确区分。

六、怎样有效修改实验

先写出预期，再一次改变一个因素；记录模型、初值、采样率、延迟、
积分步长、限幅、评价区间及单位。尤其注意“一个控制周期延迟”在
采样周期改变时也会改变实际延迟。先做FK/Jacobian/动力学/功率回代，
再讨论整条轨迹曲线，比直接调整一堆增益更容易定位原因。
