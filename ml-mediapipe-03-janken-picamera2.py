import cv2
import numpy as np
from sklearn.linear_model import Perceptron
from tensorflow import keras
import sys
from PIL import Image, ImageTk
import threading
import time
import subprocess
import tkinter as tk
    
import copy
import math
import mediapipe as mp
from picamera2 import Picamera2

# じゃんけんの手のベクトル形式を格納した配列。入力データとして用いる
# グー [1, 0, 0], チョキ [0, 1, 0], パー [0, 0, 1]
janken_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# グー, チョキ, パーの名称を格納した配列
janken_class =  ['グー', 'チョキ', 'パー']
janken_class_eng = ['Gu', 'Choki', 'Pa']

# 過去何回分の手を覚えているか
n = 3

# じゃんけんの過去の手の初期化
# 人間の手とコンピュータの手をそれぞれn回分。さらに1回分につき3個の数字が必要
Jprev = np.zeros(3*n*2) 

# 過去の手（ベクトル形式）をランダムに初期化
for i in range(2*n):
    j = np.random.randint(0, 3)
    Jprev[3*i:3*i+3] = janken_array[j]

# 現在の手（0～2の整数）をランダムに初期化
j = np.random.randint(0, 3)

# 過去の手（入力データ）をscikit_learn用の配列に変換
Jprev_set = np.array([Jprev])
# 現在の手（ターゲット）をscikit_learn用の配列に変換
jnow_set = np.array([j])

# 単純パーセプトロンを定義
clf_janken = Perceptron(random_state=None)
# ランダムな入力でオンライン学習を1回行う。
# 初回の学習では、あり得るターゲット(0, 1, 2)を分類器に知らせる必要がある
clf_janken.partial_fit(Jprev_set, jnow_set, classes=[0, 1, 2])

# 勝敗の回数を初期化
win = 0
draw = 0
lose = 0

# 状態保存用のフラグ
appliStop = False
jankenLoop = False
recognizedHand = 0

# 学習済ファイルの確認
if len(sys.argv)==2:
    savefile = sys.argv[1]
    try:
        model = keras.models.load_model(savefile)
    except IOError:
        print('学習済ファイル{0}を開けません'.format(savefile))
        sys.exit()
    except AttributeError:
        print('TensorFlow 2 で作成した学習済ファイルしか開けません')
        sys.exit()
else:
    print('使用法: python ml-10-09-janken-deep.py 学習済ファイル.h5')
    sys.exit()

# モデルロード #############################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
)


model = keras.models.load_model(savefile)

num_joints = 21
    
data = np.empty((num_joints, 2), float)

# X:画像から計算したベクトル、y:教師データ
#X = np.empty((0, num_joints*2), float) 
#y = np.array([], int)

def calc_palm_moment_data(data):  
    palm_array = np.empty((0, 2), int)
    for i in [0, 1, 5, 9, 13, 17]:
        landmark_point = [np.array((data[i, 0], data[i, 1]))]
        palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv2.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return cx, cy

def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv2.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def draw_landmarks(image, cx, cy, landmarks, handedness):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv2.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv2.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv2.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv2.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv2.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv2.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv2.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv2.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv2.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv2.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv2.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv2.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv2.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv2.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv2.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv2.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv2.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv2.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv2.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv2.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv2.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score

        cv2.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv2.putText(image, handedness.classification[0].label[0],
                   (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                   2, cv2.LINE_AA)  # label[0]:一文字目だけ

    return image

# 手の認識をし続ける関数
def imageProcessing():
    picam2 = Picamera2()
    try:
        # camver=1 or camver=2
        preview_config = picam2.create_preview_configuration({'format': 'XRGB8888', 'size': (640, 480)}, raw=picam2.sensor_modes[3])
    except IndexError:
        try:
            # camver=3
            preview_config = picam2.create_preview_configuration({'format': 'XRGB8888', 'size': (640, 480)}, raw=picam2.sensor_modes[2])
        except IndexError:
            preview_config = picam2.create_preview_configuration({'format': 'XRGB8888', 'size': (640, 480)})
    picam2.configure(preview_config)
    picam2.start()

    #cv2.namedWindow('Janken Demo', cv2.WINDOW_NORMAL)
    while True:
        # カメラキャプチャ #####################################################
        image = picam2.capture_array()
        image = cv2.flip(image, 1)  # ミラー表示
        debug_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # 検出実施 #############################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # 描画 ################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # 描画
                debug_image = draw_landmarks(debug_image, cx, cy,
                                             hand_landmarks, handedness)
        if results.multi_hand_landmarks is not None:
            image_width, image_height = debug_image.shape[1], debug_image.shape[0]
            for index, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                data[index, 0] = landmark_x
                data[index, 1] = landmark_y

            x0, y0 = calc_palm_moment_data(data.astype(int))
            for index in range(num_joints):
                data[index, 0] -= x0
                data[index, 1] -= y0
            # normalizing |moment - wrist| as 1
            length = math.sqrt(data[0, 0]**2 +  data[0, 1]**2)
            for index in range(num_joints):
                data[index, 0] /= length
                data[index, 1] /= length  

            X = np.array([data.flatten()])
            result = np.argmax(model.predict(X, verbose=0), axis=-1)
            cv2.putText(debug_image, janken_class_eng[result[0]], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
            
            # 分類結果をrecognizedHandに格納
            global recognizedHand
            recognizedHand = result[0]
        # 手と判定されている領域を表示
        cv2.imshow('Janken Demo', debug_image)
        # waitを入れる
        key = cv2.waitKey(1) 

        if appliStop == True:
            break

    cv2.destroyAllWindows()
    app.jankenStop()
    app.quit()

class Application(tk.Frame):
    # 初期化用関数
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.w = 300
        self.h = 300
        self.pack()
        self.create_widgets()
        root.protocol('WM_DELETE_WINDOW', self.close_window)
        self.recogthread = threading.Thread(target=imageProcessing)
        self.recogthread.start()

    # アプリ終了時に呼ばれる関数
    def close_window(self):
        global jankenLoop
        global appliStop 
        jankenLoop = False
        appliStop = True

    # GUI部品の初期化
    def create_widgets(self):
        w = self.w
        h = self.h

        # コンピュータの手を表示する領域を初期化
        self.comp_canvas = tk.Canvas(self, width=w, height=h, bg='white')
        self.comp_blank_img = tk.PhotoImage(width=w, height=h)
        self.comp_canvas.create_image((w/2,h/2), image=self.comp_blank_img, state='normal')
        self.comp_canvas.image = self.comp_blank_img
        self.comp_canvas.grid(row=0, column=0)

        # コンピュータの手の画像の読み込み
        self.comp_gu_img = ImageTk.PhotoImage(image=Image.open('ml-images/comp_gu.png').resize((w,h)))
        self.comp_choki_img = ImageTk.PhotoImage(image=Image.open('ml-images/comp_choki.png').resize((w,h)))
        self.comp_pa_img = ImageTk.PhotoImage(image=Image.open('ml-images/comp_pa.png').resize((w,h)))

        # 人間の手を表示する領域を初期化 
        self.human_canvas = tk.Canvas(self, width=w, height=h, bg='white')
        self.human_blank_img = tk.PhotoImage(width=w, height=h)
        self.human_canvas.create_image((w/2,h/2), image=self.human_blank_img, state='normal')
        self.human_canvas.image = self.human_blank_img
        self.human_canvas.grid(row=0, column=1)

        # 人間の手の画像の読み込み
        self.human_gu_img = ImageTk.PhotoImage(image=Image.open('ml-images/human_gu.png').resize((w,h)))
        self.human_choki_img = ImageTk.PhotoImage(image=Image.open('ml-images/human_choki.png').resize((w,h)))
        self.human_pa_img = ImageTk.PhotoImage(image=Image.open('ml-images/human_pa.png').resize((w,h)))

        # メッセージ表示領域の初期化
        self.message_canvas = tk.Canvas(self, width=2*w, height=30, bg='white')
        self.message_canvas.grid(row=1, column=0, columnspan=2)

        # 結果表示領域の初期化
        self.result_canvas = tk.Canvas(self, width=2*w, height=30, bg='white')
        self.result_canvas.grid(row=2, column=0, columnspan=2)

        # じゃんけん開始ボタンの初期化
        self.janken_btn = tk.Button(self, text='じゃんけん開始', command=self.janken_start, relief='raised')
        self.janken_btn.grid(row=3, column=0)

        # クリアボタンの初期化
        self.reset_btn = tk.Button(self, text='集計のクリア', command=self.clear)
        self.reset_btn.grid(row=3, column=1)

    # クリアボタンが押されたときに呼ばれる関数
    def clear(self):
        global draw, lose, win
        draw = 0
        lose = 0
        win = 0
        self.message_canvas.delete('all')
        result_text = 'コンピュータの勝ち: {1}, あいこ: {2}, あなたの勝ち: {0} '.format(win, lose, draw)
        self.result_canvas.delete('all')
        self.result_canvas.create_text(self.w, 15, text=result_text)

    # じゃんけんが動作中かチェックするための関数
    def jankenAlive(self):
        try:
            self.jankenthread.is_alive()
        except AttributeError:
            return False

    # じゃんけんを停止するための関数
    def jankenStop(self):
        try:
            self.jankenthread.join()
        except AttributeError:
            pass

    # じゃんけん開始ボタンが押されたときに呼ばれる関数
    def janken_start(self):
        global jankenLoop
        if jankenLoop == False:
            jankenLoop = True
            if not self.jankenAlive():
                self.jankenthread = threading.Thread(target=self.janken_loop)
                self.jankenthread.start()
            self.janken_btn.config(relief='sunken')
            self.reset_btn.config(state='disabled')
        else:
            jankenLoop = False
            self.janken_btn.config(relief='raised')
            self.reset_btn.config(state='normal')

    # じゃんけんのループ
    def janken_loop(self):
        w = self.w
        h = self.h
        global jankenLoop
        while jankenLoop == True:
            time.sleep(1)
            # メッセージ領域に「じゃんけん」と表示
            message_text = 'じゃんけん'
            self.message_canvas.delete('all')
            self.message_canvas.create_text(self.w, 15, text=message_text)
            # 「じゃんけんぽん」という音声を再生
            args = ['mpg321', '-q', 'ml-sound/jankenpon.mp3']
            try:
                process = subprocess.Popen(args).wait()
            except FileNotFoundError:
                time.sleep(2)
            # メッセージ領域に「ぽん！」と表示
            message_text = 'ぽん！'
            self.message_canvas.delete('all')
            self.message_canvas.create_text(self.w, 15, text=message_text)

            # 人間の手を画像処理の結果から決定
            j = recognizedHand
            global Jprev
            # 過去のじゃんけんの手（ベクトル形式）をscikit_learn形式に
            Jprev_set = np.array([Jprev])
            # 現在のじゃんけんの手（0～2の整数）をscikit_learn形式に
            jnow_set = np.array([j])

            # コンピュータが、過去の手から人間の現在の手を予測
            jpredict = clf_janken.predict(Jprev_set)

            # 人間の手
            your_choice = j
            # 予測を元にコンピュータが決めた手
            # 予測がグーならパー、予測がチョキならグー、予測がパーならチョキ
            comp_choice = (jpredict[0] + 2)%3

            if comp_choice == 0:
                # コンピュータのグー画像表示
                self.comp_canvas.create_image((w/2,h/2), image=self.comp_gu_img, state='normal')
                self.comp_canvas.image = self.comp_gu_img
            elif comp_choice == 1:
                # コンピュータのチョキ画像表示
                self.comp_canvas.create_image((w/2,h/2), image=self.comp_choki_img, state='normal')
                self.comp_canvas.image = self.comp_choki_img
            else:
                # コンピュータのパー画像表示
                self.comp_canvas.create_image((w/2,h/2), image=self.comp_pa_img, state='normal')
                self.comp_canvas.image = self.comp_pa_img

            if your_choice == 0:
                # 人間のグー画像表示
                self.human_canvas.create_image((w/2,h/2), image=self.human_gu_img, state='normal')
                self.human_canvas.image = self.human_gu_img
            elif your_choice == 1:
                # 人間のチョキ画像表示
                self.human_canvas.create_image((w/2,h/2), image=self.human_choki_img, state='normal')
                self.human_canvas.image = self.human_choki_img
            else:
                # 人間のパー画像表示
                self.human_canvas.create_image((w/2,h/2), image=self.human_pa_img, state='normal')
                self.human_canvas.image = self.human_pa_img

            # 勝敗結果を更新
            global draw, lose, win
            if your_choice == comp_choice:
                draw += 1
            elif your_choice == (comp_choice+1)%3:
                lose += 1
            else:
                win += 1

            # 勝敗結果を表示
            result_text = 'コンピュータの勝ち: {1}, あいこ: {2}, あなたの勝ち: {0} '.format(win, lose, draw)
            self.result_canvas.delete('all')
            self.result_canvas.create_text(self.w, 15, text=result_text)

            # 過去の手（入力データ）と現在の手（ターゲット）とでオンライン学習
            clf_janken.partial_fit(Jprev_set, jnow_set)

            # 過去の手の末尾に現在のコンピュータの手を追加
            Jprev = np.append(Jprev[3:], janken_array[comp_choice])
            # 過去の手の末尾に現在の人間の手を追加
            Jprev = np.append(Jprev[3:], janken_array[your_choice])

root = tk.Tk()
app = Application(master=root)
app.master.title('MediaPipe じゃんけん')
app.mainloop()
