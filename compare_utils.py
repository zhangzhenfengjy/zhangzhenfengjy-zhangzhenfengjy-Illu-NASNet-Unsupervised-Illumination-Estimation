def writeErrToScreen(errorName, errorArr, epoch, j):
    print(('[%d/%d] {0}:' % (epoch, j)).format(errorName), end=' ')
    print('%.6f' % errorArr.data.item(), end=' ')
    print('.')

def writeErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:'% (epoch, j) ).format(errorName) )
    fileOut.write('%.6f ' % errorArr.data.item() )
    fileOut.write('.\n')

def writewhdrToFile(Name, whdr, fileOut, avgwhdr):
    fileOut.write('{0}      '.format(Name))
    fileOut.write('%.6f     ' % whdr)
    fileOut.write('%.6f     ' % avgwhdr)
    fileOut.write('\n')

def writeLossToFile(Loss1, Loss1_value, Loss2, Loss2_value, Total_Loss, Total_Loss_value, fileOut, epoch, j):
    fileOut.write('[%d/%d]    :' % (epoch, j))
    fileOut.write('{0}:'.format(Loss1))
    fileOut.write('%.6f ' % Loss1_value)
    fileOut.write('{0}:'.format(Loss2))
    fileOut.write('%.6f ' % Loss2_value)
    fileOut.write('{0}:'.format(Total_Loss))
    fileOut.write('%.6f ' % Total_Loss_value)
    fileOut.write('\n')

def writeNpErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    print('%.6f' % errorArr, end = ' ')
    print('.')

def writeNpErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName) )
    fileOut.write('%.6f ' % errorArr)
    fileOut.write('.\n')

import numpy as np
def turnErrorIntoNumpy(errorArr):
    errorNp = []
    errorNp.append(errorArr.data.item() )
    return np.array(errorNp)[np.newaxis, :]



def compute_whdr(reflectance, judgements, delta=0.1):
    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    error_equal_sum = 0.0
    error_inequal_sum = 0.0

    weight_sum = 0.0
    weight_equal_sum = 0.0
    weight_inequal_sum = 0.0

    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0.0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker == 'E':
            if darker != alg_darker:
                error_equal_sum += weight
            weight_equal_sum += weight
        else:
            if darker != alg_darker:
                error_inequal_sum += weight
            weight_inequal_sum += weight

        if darker != alg_darker:
            error_sum += weight
        weight_sum += weight

    if weight_sum:
        return (error_sum / weight_sum), error_equal_sum/( weight_equal_sum + 1e-10), error_inequal_sum/(weight_inequal_sum + 1e-10)
    else:
        return None



