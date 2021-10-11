def main():
    import cv2
    import pygame
    import numpy as np
    from tensorflow import keras
    import numpy as np
    import keyboard

    model = keras.models.load_model("12class_v4_simple_128.h5")
    game = 1
    winner = 0
    def RGB_to_gray(rgb, judge):
        img = rgb
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
        if judge == 1:
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
            img[:,:,0] = (0* r + 0* g + 0 * b)
            img[:,:,1] = (0* r + 0* g + 0 * b)
            img[:, :, 2] = gray
        elif judge == 2:
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
            img[:,:,0] = gray
            img[:,:,1] = (0* r + 0* g + 0 * b)
            img[:, :, 2] = (0* r + 0 * g + 0* b)
        elif judge == 3:
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
            img[:,:,0] = gray
            img[:,:,1] = (0* r + 0* g + 0 * b)
            img[:, :, 2] =  gray
        return img
    gamestart = 0
    
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
    skill_to_attack = {1:10, 2:50, 3:40}
    my_font = pygame.font.Font("ShangShouZhiZunShuFaTi-2.ttf", 200)
    skill_font = pygame.font.Font("ShangShouZhiZunShuFaTi-2.ttf", 150)
    list_font = pygame.font.Font("ShangShouZhiZunShuFaTi-2.ttf", 80)

    def return_font(predict_class, judge_win):
        if judge_win == 0:
            text_surface = my_font.render(num_to_word[predict_class], True, (0,0,0))
        else:
            text_surface = my_font.render(' ', True, (0,0,0))
        return text_surface
    def return_skill_font(player_list):
        judge = 0
        if player_list == [1,3,8]:
            text_surface = skill_font.render('千鸟', True, (255,255,255))
            judge = 1
        elif player_list == [5,7,8,11,6,2]:
            text_surface = skill_font.render('火遁·大火球之术', True, (255,255,255))
            judge = 2
        elif player_list == [7, 6, 4, 2]:
            text_surface = skill_font.render('土遁·土龙弹', True, (255,255,255))
            judge = 3
        else:
            text_surface = skill_font.render('', True, (0,0,0))
        textpos = text_surface.get_rect(center=(482,360))
        return text_surface, textpos, judge
    
    def return_list_font(player_list, judge_win):
        t = ''
        if judge_win == 0:
            for i in player_list:
                t += num_to_word[i]
            text_surface = list_font.render(t, True, (0,0,0))
        else:
            text_surface = list_font.render('', True, (0,0,0))
        return text_surface
    
    def return_blood_font(player_blood, judge_win):
        if judge_win == 0:
            t = '血量: ' + player_blood
            text_surface = list_font.render(t, True, (255,255,255))
        else:
            text_surface = list_font.render(' ', True, (255,255,255))
        return text_surface
    def return_player_font(judge_winner):
        if judge_winner != 0:
            t = '玩家' + str(judge_winner)
            text_surface = list_font.render(t, True, (255,255,255))
        else:
            text_surface = list_font.render(' ', True, (255,255,255))
        return text_surface
        
    def predict_class(x, treshold = 0.82):
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
    player_list_1 = []
    pre_class_1 = [12,1]
    player_list_2 = []
    pre_class_2 = [12,1]
    skill_open = skill_length
    judge_win = 0
    judge_winer = 0
    player_blood_1 = 100
    player_blood_2 = 100
    
    while True:
        while game == 1:
            surface.fill([0,0,0])

            ret ,frame = cap.read()
            # print(ret)
            frame=cv2.flip(frame,1)
            color = (0, 255, 255)
            if judge_win == 0:
                cv2.rectangle(frame, (40, 220), (221, 401), color, 2)
                cv2.rectangle(frame, (420, 220), (601, 401), color, 2)
            k=cv2.waitKey(1)

            testframe = frame
            grayImage_1 = cv2.cvtColor(frame[222:400,42:220], cv2.COLOR_BGR2GRAY)
            grayImage_2 = cv2.cvtColor(frame[222:400,422:600], cv2.COLOR_BGR2GRAY)

            grayImage_1 = cv2.resize(grayImage_1, (128, 128), interpolation=cv2.INTER_AREA)
            grayImage_2 = cv2.resize(grayImage_2, (128, 128), interpolation=cv2.INTER_AREA)

            grayImage_1 = np.expand_dims(grayImage_1, axis=-1)
            grayImage_2 = np.expand_dims(grayImage_2, axis=-1)
            x_1 = np.expand_dims(grayImage_1, axis=0)
            x_2 = np.expand_dims(grayImage_2, axis=0)
            y_pre_1 = predict_class(x_1)
            y_pre_2 = predict_class(x_2)

            pre_class_1 = update_pre_class(pre_class_1, y_pre_1)
            pre_class_2 = update_pre_class(pre_class_2, y_pre_2)

            if(pre_class_1[0] != 12 and pre_class_1[1] >= 7):
                if(player_list_1 == [] or player_list_1[-1] != pre_class_1[0]):
                    player_list_1.append(pre_class_1[0])
                    pre_class_1 = [12, 0]
            elif(pre_class_1[0] == 12 and pre_class_1[1] >= 20):
                pre_class_1 = [12, 0]
                player_list_1 = []

            if(pre_class_2[0] != 12 and pre_class_2[1] >= 7):
                if(player_list_2 == [] or player_list_2[-1] != pre_class_2[0]):
                    player_list_2.append(pre_class_2[0])
                    pre_class_2 = [12, 0]
            elif(pre_class_2[0] == 12 and pre_class_2[1] >= 20):
                pre_class_2 = [12, 0]
                player_list_2 = []

            testframe = np.fliplr(testframe)
            testframe = np.rot90(testframe)
            testframe = cv2.cvtColor(testframe, cv2.COLOR_BGR2RGB)
            testframe = cv2.resize(testframe,(720, 965))
            
            text_surface_1, textpos_1, judge_1 = return_skill_font(player_list_1)
            text_surface_2, textpos_2, judge_2 = return_skill_font(player_list_2)
            judge_win = 0
            judge_winer = 0
            if judge_1 or judge_2:
                skill_open = skill_open - 1  
                if judge_1:
                    judge_winer = 1
                    player_list_2 = []
                    judge_win = judge_1
                    text_surface, textpos = text_surface_1, textpos_1 
                else:
                    judge_winer = 2
                    player_list_1 = []
                    judge_win = judge_2
                    text_surface, textpos = text_surface_2, textpos_2 
            else:
                text_surface, textpos = text_surface_1, textpos_1


            if skill_open <= (skill_length - 1):
                testframe = RGB_to_gray(testframe,judge_win)
                skill_open -= 1
                if skill_open == 0:
                    skill_open = skill_length
                    if judge_winer == 1:
                        player_blood_2 -= skill_to_attack[judge_win]
                    elif judge_winer == 2:
                        player_blood_1 -= skill_to_attack[judge_win]
                    player_list_1 = []
                    player_list_2 = []
            # print(skill_open)
            surf = pygame.surfarray.make_surface(testframe)

            if(player_blood_1 <= 0):
                game += 1
                winner = 2
            elif(player_blood_2 <= 0):
                game += 1
                winner = 1

            surface.blit(surf, (0,0))
            surface.blit(return_font(y_pre_1, judge_win), (100,100))
            surface.blit(return_font(y_pre_2, judge_win), (500,100))
            surface.blit(return_blood_font(str(player_blood_1), judge_win), (0,600))
            surface.blit(return_blood_font(str(player_blood_2), judge_win), (550,600))
            surface.blit(return_list_font(player_list_1, judge_win), (0,0))
            surface.blit(return_list_font(player_list_2, judge_win), (450,0))
            surface.blit(return_player_font(judge_winer), (200,100))
            surface.blit(text_surface, textpos)
            pygame.display.flip()
    
        while game == 2:
            surface.fill([0,0,0])

            ret ,frame = cap.read()
            # print(ret)
            frame=cv2.flip(frame,1)
            color = (0, 255, 255)
            k=cv2.waitKey(1)

            testframe = frame
            testframe = np.fliplr(testframe)
            testframe = np.rot90(testframe)
            testframe = cv2.cvtColor(testframe, cv2.COLOR_BGR2RGB)
            testframe = cv2.resize(testframe,(720, 965))
            surf = pygame.surfarray.make_surface(testframe)
            surface.blit(surf, (0,0))
            text_surface = skill_font.render('玩家'+str(winner)+' 胜', True, (0,0,0))
            textpos = text_surface.get_rect(center=(482,360))
            surface.blit(text_surface, textpos)
            pygame.display.flip()


if __name__ == '__main__':    
    main()