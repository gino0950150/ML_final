def main():
    import cv2
    import pygame
    import numpy as np
    import tensorflow as tf
    import numpy as np
    def RGB_to_gray(rgb, judge):
        img = rgb
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
        if judge == 1:
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
            img[:,:,0] = gray
            img[:,:,1] = gray
            img[:, :, 2] = gray
        elif judge == 2:
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
            img[:,:,0] = gray
            img[:,:,1] = (0* r + 0* g + 0 * b)
            img[:, :, 2] = (0* r + 0 * g + 0* b)
        return img
    model = tf.keras.models.load_model("models/12class_v4_simple_128.h5")
    pygame.init()
    pygame.display.set_caption("OpenCV camera stream on Pygame")
    surface = pygame.display.set_mode((965, 720))
    #0 Is the built in camera
    cap = cv2.VideoCapture(0)
    #Gets fps of your camera
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)
    #If your camera can achieve 60 fps
    #Else just have this be 1-30 fps
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    num_to_word = {0:'子', 1:'丑', 2:'寅', 3:'卯', 4:'辰', 5:'巳', 6:'午', 7:'未', 8:'申', 9:'酉', 10:'戌', 11:'亥', 12:' '}
    my_font = pygame.font.Font("ShangShouZhiZunShuFaTi-2.ttf", 200)
    skill_font = pygame.font.Font("ShangShouZhiZunShuFaTi-2.ttf", 400)
    list_font = pygame.font.Font("ShangShouZhiZunShuFaTi-2.ttf", 100)

    def return_font(predict_class):
        text_surface = my_font.render(num_to_word[predict_class], True, (0,0,0))
        return text_surface
    def return_skill_font(player_list):
        judge = 0
        if player_list == [1,3]:
            text_surface = skill_font.render('千鸟', True, (0,0,0))
            judge = 1
        elif player_list == [3,1]:
            text_surface = skill_font.render('火盾', True, (0,0,0))
            judge = 2
        else:
            text_surface = skill_font.render('', True, (0,0,0))
        textpos = text_surface.get_rect(center=(482,360))
        return text_surface, textpos, judge
    def return_list_font(playyer_list):
        t = ''
        for i in player_list:
            t += num_to_word[i]
        text_surface = list_font.render(t, True, (0,0,0))
        return text_surface

    def predict_class(x, treshold = 0.99):
        matrix = model.predict(x)
        y_pre = np.argmax(model.predict(x), axis=-1)
        if(matrix[0][y_pre] >= treshold):
            return y_pre[0]
        else:
            return 12
    
    def update_pre_class(l, c):
        if c == l[0]:
            return [l[0], l[1] + 1]
        else : 
            return [c, 0]



    skill_length = 30
    player_list = []
    pre_class = [12,1]
    skill_open = skill_length

    while True:
        surface.fill([0,0,0])

        ret ,frame = cap.read()
        # print(ret)
        frame=cv2.flip(frame,1)
        color = (0, 255, 255)
        
        cv2.rectangle(frame, (190, 210), (449, 469), color, 2)
        k=cv2.waitKey(1)

        # print(frame[212:468,192:448].shape) 256 256 3
        testframe = frame
        grayImage = cv2.cvtColor(frame[212:468,192:448], cv2.COLOR_BGR2GRAY)
        grayImage = cv2.resize(grayImage, (128, 128), interpolation=cv2.INTER_AREA)
        grayImage = np.expand_dims(grayImage, axis=-1)
        x = np.expand_dims(grayImage, axis=0)
        y_pre = predict_class(x)

        pre_class = update_pre_class(pre_class, y_pre)
        if(pre_class[0] != 12 and pre_class[1] >= 8):
            if(player_list == [] or player_list[-1] != pre_class[0]):
                player_list.append(pre_class[0])
                pre_class = [12, 0]
        elif(pre_class[0] == 12 and pre_class[1] >= 75):
            pre_class = [12, 0]
            player_list = []

        testframe = np.fliplr(testframe)
        testframe = np.rot90(testframe)
        testframe = cv2.cvtColor(testframe, cv2.COLOR_BGR2RGB)
        testframe = cv2.resize(testframe,(720, 965))

        text_surface, textpos, judge = return_skill_font(player_list) 

        if judge:
            skill_open = skill_open - 1  
        if skill_open <= (skill_length - 1):
            testframe = RGB_to_gray(testframe,judge)
            skill_open -= 1
            if skill_open == 0:
                skill_open = skill_length
                player_list = []
        # print(skill_open)
        surf = pygame.surfarray.make_surface(testframe)

        surface.blit(surf, (0,0))
        surface.blit(return_font(y_pre), (100,100))
        surface.blit(return_list_font(player_list), (0,0))
        surface.blit(text_surface, textpos)

        pygame.display.flip()

if __name__ == '__main__':    
    main()