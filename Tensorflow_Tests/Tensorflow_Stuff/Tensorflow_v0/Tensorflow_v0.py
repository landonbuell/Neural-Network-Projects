"""
Landon Buell
Up & Running Tensorflow
Geron, Ch.9
21 May 2020
"""

            #### Imports ####

import tensorflow as tf

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    x = tf.Variable(3,name='x')
    y = tf.Variable(4,name='y')

    f = x*x*y + y + 2

    # Open TF Session
    my_session = tf.Session()
    
    my_session.run(x.initializer)
    my_session.run(y.initializer)

    result = my_session.run(f)

    print(result)