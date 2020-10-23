# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:52:31 2020

@author: micha
"""
import csv
import random
import pickle

def make_neg_samples():
    base = "notouch_face_single_"
    for i in range(1,11):
        pass
    
def parse_data(filename):

    rssi = []
    accel = []
    mag = []
    gyro = []
    rssi_i = []
    i = 0
    
    with open(filename + ".csv", newline='') as file:
        divisor = sum(1 for i in csv.reader(file, delimiter=','))/1000
        file.close()

    with open(filename + ".csv", newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:

            if len(row) == 1 or len(row) == 2:  # 1 column format
                if row[0][0] == "R":  # RSSI
                    rssi.append(float(row[1]))
                    rssi_i.append(int(i/divisor))

                elif row[0][0] == "A":  # Acceleration
                    acc_i = row[0].find(":")
                    values = row[0].split(",")
                    accel.append([float(values[0][acc_i+3:]), float(values[1]), float(values[2][:-1])])

                elif row[0][0] == "M":  # Magnetometer
                    mag_i = row[0].find(":")
                    values = row[0].split(",")
                    mag.append([float(values[0][mag_i+3:]), float(values[1]), float(values[2][:-1])])

                elif row[0][0] == "G":  # Gyroscope
                    gyro_i = row[0].find(":")
                    values = row[0].split(",")
                    gyro.append([float(values[0][gyro_i+3:]), float(values[1]), float(values[2][:-1])])

            else:

                if row[0][0] == "D":  # RSSI
                    rssi_i = row[0].find("RSSI = ")
                    rssi.append(float(row[0][rssi_i+7:]))

                elif row[0][0] == "A":  # Acceleration
                    acc_i = row[0].find(":")
                    accel.append([float(row[0][acc_i+3:]), float(row[1]), float(row[2][:-1])])

                elif row[0][0] == "M":  # Magnetometer
                    mag_i = row[0].find(":")
                    mag.append([float(row[0][mag_i+3:]), float(row[1]), float(row[2][:-1])])

                elif row[0][0] == "G":  # Gyroscope
                    gyro_i = row[0].find(":")
                    gyro.append([float(row[0][gyro_i+3:]), float(row[1]), float(row[2][:-1])])
            i += 1
        return rssi, accel, mag, gyro, rssi_i
    
def make_samples(data, label, window=5, num_samples=1000, num_around=10, norm_std=10):
    """
    Return input size needed for NN
    """
    #9 samples per reading, window rssi_vals, and num_around for each rssi val
    size = window + window*num_around*9
    
    rssi, accel, mag, gyro, rssi_i = data
    data = []
    rssi_vals = []
    
    #Make copies with a sliding window
    offset = 0
    while offset + window < len(rssi) + 1:
        #save corresponding location in data
        rssi_vals.append([(rssi[offset+i], rssi_i[offset+i]) for i in range(window)])
        offset += 1
    
    #Duplicate and perturb all of the copies
    while len(data) < num_samples:
        new_rssi = random.choice(rssi_vals).copy()
        #perturb (2=hyperparameter)
        new_obs = [rssi[0] + random.gauss(0,2) for rssi in new_rssi]
        #perturb
        for rssi in new_rssi:
            #take num_around random samples around each rssi point
            indices = sorted([round(rssi[1] + random.gauss(0,norm_std)) for i in range(num_around)])
            etc_samples = []
            for i in indices:
                etc_samples += accel[i]
                etc_samples += mag[i]
                etc_samples += gyro[i]
            new_obs += etc_samples
        data.append(new_obs)
    
    return data, [(label,) for i in range(num_samples)]


def make_neg_samples(data, label, window=5, num_samples=1000, num_around=10, norm_std=10):
    """
    Return input size needed for NN
    """
    #9 samples per reading, window rssi_vals, and num_around for each rssi val
    size = window + window*num_around*9
    
    rssi, accel, mag, gyro, rssi_i = data
    data = []
    rssi_vals = []
    
    #Make copies with a sliding window
    offset = 0
    while offset + window < len(rssi) + 1:
        #save corresponding location in data
        rssi_vals.append([(rssi[offset+i], rssi_i[offset+i]) for i in range(window)])
        offset += 1
    
    
    #First make sure every copy is included
    for new_rssi in rssi_vals:
        #perturb (2=hyperparameter)
        new_obs = [rssi[0] for rssi in new_rssi]
        #perturb
        for rssi in new_rssi:
            #take num_around random samples around each rssi point
            indices = sorted([round(rssi[1] + random.gauss(0,norm_std)) for i in range(num_around)])
            etc_samples = []
            for i in indices:
                etc_samples += accel[i]
                etc_samples += mag[i]
                etc_samples += gyro[i]
            new_obs += etc_samples
        data.append(new_obs)
    
    #Duplicate and perturb all of the copies
    while len(data) < num_samples:
        new_rssi = random.choice(rssi_vals).copy()
        #perturb (2=hyperparameter)
        new_obs = [rssi[0] + random.gauss(0,2) for rssi in new_rssi]
        #perturb
        for rssi in new_rssi:
            #take num_around random samples around each rssi point
            indices = sorted([round(rssi[1] + random.gauss(0,norm_std)) for i in range(num_around)])
            etc_samples = []
            for i in indices:
                etc_samples += accel[i]
                etc_samples += mag[i]
                etc_samples += gyro[i]
            new_obs += etc_samples
        data.append(new_obs)
    
    return data, [(label,) for i in range(len(data))]

if __name__ == "__main__":
    #original working NN used make_samples(parse_data(base+notouch+str(i)), 0, window=12, num_samples=100, num_around=3)
    
    # OLD DATA
    random.seed(6.808)
    base = "data_motions/"
    notouch = "notouch_face_single_"
    touch = "touch_face_single"
    data = []
    labels = []
    for i in range(1, 11):
        #negative examples (2 has insufficient rssi values)
        if i != 2:
            new_data, new_labels = make_samples(parse_data(base+notouch+str(i)), 0, window=15, num_samples=100, num_around=3)
            data += new_data
            labels += new_labels
        #positive examples (i has insufficient rssi values)
        if i != 1:
            new_data, new_labels = make_samples(parse_data(base+touch+str(i)), 1, window=15, num_samples=40, num_around=3)
            data += new_data
            labels += new_labels
            
    # NEW DATA
    base = "data_motions/newdata/"
    notouch = "negm_single_may5"
    touch = "m_single_may5"
    
    #negative examples
    new_data, new_labels = make_neg_samples(parse_data(base+notouch), 0, window=15, num_samples=3000, num_around=3)
    data += new_data
    labels += new_labels
    
    for i in range(1, 21):
        #positive examples
        new_data, new_labels = make_samples(parse_data(base+str(i)+touch), 1, window=15, num_samples=100, num_around=3)
        data += new_data
        labels += new_labels
    
            
    # shuffle examples
    indices = list(range(len(data)))
    random.shuffle(indices)
    new_ordering = indices
    data_shuffled = [0] * len(data)
    labels_shuffled = [0]*len(data)
    for i in range(len(data)):
        data_shuffled[new_ordering[i]] = data[i]
        labels_shuffled[new_ordering[i]] = labels[i]
    print(len(data_shuffled), len(labels_shuffled), "1%:" + str(labels.count((1,))/len(labels)))
    store_data = (data_shuffled, labels_shuffled)
    file = open('all_data_processed_20_large_3.pkl', 'wb')
    pickle.dump(store_data, file)
    file.close()
        
    