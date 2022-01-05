import logging

def test_log():
    logging.info('test3')


if __name__=='__main__':
    logging.basicConfig(filename='./1.log', filemode='a', level=logging.INFO)
    logging.info('success!')
    logging.info('test1')
    logging.info('test2')
    epoch = 10
    logging.info('Saving state, epoch: %d' % (epoch + 1)) 
    test_log()
