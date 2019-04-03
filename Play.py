import numpy as np
from Train import create_model_Conv2D
from Prepare import model_def, resize_image
from PIL import ImageGrab
import win32com.client as comclt


class Actor(object):
    
    def __init__(self):
        self.model = create_model_Conv2D(m_model_def.INPUT_SHAPE, m_model_def.OUT_SHAPE)
        self.model.load_weights(m_model_def.weights_file)
        print('Weights loaded : ', m_model_def.weights_file)

    def get_action(self, obs):
        vec = resize_image(obs)
        new_obs = np.expand_dims(vec, 0)
        joystick = self.model.predict(new_obs, batch_size=1)[0]
        j = np.argmax(joystick)
        print(str(j), np.uint8(joystick * 100.))
        return j


if __name__ == '__main__':
    wsh = comclt.Dispatch("WScript.Shell")
    m_model_def = model_def()
    actor = Actor()
    num_step = 10000
    try:
        for step in range(num_step):
            obs = np.array(ImageGrab.grab())
            action = actor.get_action(obs)
            for _ in range(1):
                if action == 0:
                    pass
                elif action == 1:
                    wsh.SendKeys('{UP}')
                elif action == 2:
                    wsh.SendKeys('{DOWN}')
    except:
        pass

    print('Play finished.')

