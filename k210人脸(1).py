import sensor,image,lcd  # import 相关库
import KPU as kpu
import time
from Maix import FPIOA,GPIO
from fpioa_manager import fm
from machine import UART

task_fd = kpu.load(0x100000) # 从flash 0x200000 加载人脸检测模型
task_ld = kpu.load(0x200000) # 从flash 0x300000 加载人脸五点关键点检测模型
task_fe = kpu.load(0x300000) # 从flash 0x400000 加载人脸196维特征值模型

#从SD卡中加载模型
#task_fd = kpu.load("/sd/FD_face.smodel") # 加载人脸检测模型
#task_ld = kpu.load("/sd/KP_face.smodel") # 加载人脸五点关键点检测模型
#task_fe = kpu.load("/sd/FE_face.smodel") # 加载人脸196维特征值模型

clock = time.clock()  # 初始化系统时钟，计算帧率
key_pin=16 # 设置按键引脚 FPIO16
fpioa = FPIOA()
fpioa.set_function(key_pin,FPIOA.GPIO7)
key_gpio=GPIO(GPIO.GPIO7,GPIO.IN)
last_key_state=1
key_pressed=0 # 初始化按键引脚 分配GPIO7 到 FPIO16
def check_key(): # 按键检测函数，用于在循环中检测按键是否按下，下降沿有效
    global last_key_state
    global key_pressed
    val=key_gpio.value()
    if last_key_state == 1 and val == 0:
        key_pressed=1
    else:
        key_pressed=0
    last_key_state = val


#####################配置串口#####################
fm.register(6, fm.fpioa.UART1_RX, force=True)
fm.register(7, fm.fpioa.UART1_TX, force=True)

uart = UART(UART.UART1, 9600, 8, 1, 0, timeout=1000, read_buf_len=4096)
uart.read()

names = [] # 人名标签，与上面列表特征值一一对应。
record_ftrs=[] #空列表 用于存储按键记录下人脸特征， 可以将特征以txt等文件形式保存到sd卡后，读取到此列表，即可实现人脸断电存储。

################## 开机时读取SD卡中的人脸信息 ##################
with open("/sd/faceinfo.txt", "r") as f:
    while(1):
        lin = f.readline()  # 按行读取 录入时的时候是按行录入的
        if not lin:
            break
        stu_num = lin[0:lin.index('#')]    #获取学号
        names.append(stu_num)              #追加到学号列表
        faceftr = lin[lin.index('#')+1:]   #截取后半段的人脸特征
        record_ftrs.append(eval(faceftr))  #向人脸特征列表中添加SD卡中的已存特征
        print("%s : %s" % (stu_num,faceftr))
    f.close()


print(names)
print(record_ftrs)

lcd.init() # 初始化lcd
lcd.rotation(2)
sensor.reset() #初始化sensor 摄像头
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(1) #设置摄像头镜像
sensor.set_vflip(1)   #设置摄像头翻转
sensor.run(1) #使能摄像头
anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025) #anchor for face detect 用于人脸检测的Anchor
dst_point = [(44,59),(84,59),(64,82),(47,105),(81,105)] #standard face key point position 标准正脸的5关键点坐标 分别为 左眼 右眼 鼻子 左嘴角 右嘴角
a = kpu.init_yolo2(task_fd, 0.5, 0.3, 5, anchor) #初始化人脸检测模型
img_lcd=image.Image() # 设置显示buf
img_face=image.Image(size=(128,128)) #设置 128 * 128 人脸图片buf
a=img_face.pix_to_ai() # 将图片转为kpu接受的格式
record_ftr=[] #空列表 用于存储当前196维特征

luru_flag = 0  # 1：表示已经收到ID，准备录入
luru_name = '' # 待录入的ID

uart.read()

while(1): # 主循环
    check_key() #按键检测
    img = sensor.snapshot() #从摄像头获取一张图片
    clock.tick() #记录时刻，用于计算帧率

    #a = img.draw_string(0,50, b'Start %s'%stu_num, color=(0,255,0),scale=1.6,mono_space=1) #显示开始录入
    text = uart.read()
    if text != None and len(text) >= 2:  # 说明读到了串口数据
        print(text)
        # 人脸注册命令
        if 'lu' in text :
            print("--------人脸注册--------")
            luru_name = text[2:].decode('utf-8')  # 截取出学号
            print('学号: ', luru_name)
            if luru_name in names:  # 如果录入的ID已存在，也不再次录入
                img.draw_rectangle((90, 85, 140, 70), fill=True, color=(0, 0, 255))
                img.draw_string(110, 112, "id exist", color=(255, 255, 255), scale=1.5, mono_space=0)
                lcd.display(img)
                time.sleep(2)
                continue
            luru_flag = 1
        # 人脸清空命令
        if 'all' in text:
            names.clear()
            record_ftrs.clear()
            # 执行清空txt代码
            file_new = open('/sd/faceinfo.txt', 'w')
            file_new.write('')
            file_new.close()
            img.draw_rectangle((90, 85, 140, 70), fill=True, color=(0, 0, 255))
            img.draw_string(110, 112, "delete all", color=(255, 255, 255), scale=1.5, mono_space=0)
            lcd.display(img)
            time.sleep(2)

    # 人脸录入LCD提示
    if luru_flag == 1:
        a = img.draw_string(0,50, b'Start %s'%luru_name, color=(0,255,0),scale=1.6,mono_space=1) #显示开始录入

    try:
        code = kpu.run_yolo2(task_fd, img) # 运行人脸检测模型，获取人脸坐标位置
        if code: # 如果检测到人脸
            for i in code: # 迭代坐标框
                # Cut face and resize to 128x128
                a = img.draw_rectangle(i.rect()) # 在屏幕显示人脸方框
                face_cut=img.cut(i.x(),i.y(),i.w(),i.h()) # 裁剪人脸部分图片到 face_cut
                face_cut_128=face_cut.resize(128,128) # 将裁出的人脸图片 缩放到128 * 128像素
                a=face_cut_128.pix_to_ai() # 将猜出图片转换为kpu接受的格式
                #a = img.draw_image(face_cut_128, (0,0))
                # Landmark for face 5 points
                fmap = kpu.forward(task_ld, face_cut_128) # 运行人脸5点关键点检测模型
                plist=fmap[:] # 获取关键点预测结果
                le=(i.x()+int(plist[0]*i.w() - 10), i.y()+int(plist[1]*i.h())) # 计算左眼位置， 这里在w方向-10 用来补偿模型转换带来的精度损失
                re=(i.x()+int(plist[2]*i.w()), i.y()+int(plist[3]*i.h())) # 计算右眼位置
                nose=(i.x()+int(plist[4]*i.w()), i.y()+int(plist[5]*i.h())) #计算鼻子位置
                lm=(i.x()+int(plist[6]*i.w()), i.y()+int(plist[7]*i.h())) #计算左嘴角位置
                rm=(i.x()+int(plist[8]*i.w()), i.y()+int(plist[9]*i.h())) #右嘴角位置
                a = img.draw_circle(le[0], le[1], 4)
                a = img.draw_circle(re[0], re[1], 4)
                a = img.draw_circle(nose[0], nose[1], 4)
                a = img.draw_circle(lm[0], lm[1], 4)
                a = img.draw_circle(rm[0], rm[1], 4) # 在相应位置处画小圆圈
                # align face to standard position
                src_point = [le, re, nose, lm, rm] # 图片中 5 坐标的位置
                T=image.get_affine_transform(src_point, dst_point) # 根据获得的5点坐标与标准正脸坐标获取仿射变换矩阵
                a=image.warp_affine_ai(img, img_face, T) #对原始图片人脸图片进行仿射变换，变换为正脸图像
                a=img_face.ai_to_pix() # 将正脸图像转为kpu格式
                #a = img.draw_image(img_face, (128,0))
                del(face_cut_128) # 释放裁剪人脸部分图片
                # calculate face feature vector
                fmap = kpu.forward(task_fe, img_face) # 计算正脸图片的196维特征值
                feature=kpu.face_encode(fmap[:]) #获取计算结果  字节数组
                reg_flag = False
                scores = [] # 存储特征比对分数
                for j in range(len(record_ftrs)): #迭代已存特征值
                    score = kpu.face_compare(record_ftrs[j], feature) #计算当前人脸特征值与已存特征值的分数
                    scores.append(score) #添加分数总表
                max_score = 0
                index = 0
                for k in range(len(scores)): #迭代所有比对分数，找到最大分数和索引值
                    if max_score < scores[k]:
                        max_score = scores[k]
                        index = k
                if max_score > 80: # 如果最大分数大于80， 可以被认定为同一个人
                    uart.write(('succeed'+names[index]+'*').encode('ascii'))
                    a = img.draw_string(i.x(),i.y(), ("%s :%2.1f" % (names[index], max_score)), color=(0,255,0),scale=2) # 显示人名 与 分数
                else:
                    uart.write(('fail*').encode('ascii'))
                    a = img.draw_string(i.x(),i.y(), ("X :%2.1f" % (max_score)), color=(255,0,0),scale=2) #显示未知 与 分数

                if key_pressed == 1 and luru_flag == 1: #如果检测到按键 and 此时也收到了录入ID
                    if max_score > 80:   # 如果当前人脸置信度大于80，就不再次录入该人
                        img.draw_rectangle((90, 85, 140, 70), fill=True, color=(0, 0, 255))
                        img.draw_string(110, 112, "face exist", color=(255, 255, 255), scale=1.5, mono_space=0)
                        lcd.display(img)
                        time.sleep(2)
                        continue
                    key_pressed = 0 #重置按键状态
                    luru_flag = 0
                    record_ftr = feature
                    # 将人脸ID和特征值保存进SD卡的faceinfo.txt
                    try:
                        with open("/sd/faceinfo.txt", "a") as f:
                            f.write(luru_name+'#'+str(feature))   # 按行写入txt文件
                            f.write("\n")
                            f.close()
                    except:
                        print('保存失败')
                        continue
                    names.append(luru_name)  # 添加ID
                    record_ftrs.append(feature)  # 添加特征值
                break
    except:
        pass

    a = lcd.display(img) #刷屏显示
    #kpu.memtest()

#a = kpu.deinit(task_fe)
#a = kpu.deinit(task_ld)
#a = kpu.deinit(task_fd)
