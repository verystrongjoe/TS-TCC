import sys

if __name__ == '__main__':

    datasets = [
        'ArticularyWordRecognition',
        'AtrialFibrilation',
        'BasicMotions',
        'CharacterTrajectories',
        'Cricket',
        'DuckDuckGeese',
        'EigenWorms',
        'Epilepsy',
        'ERing',
        'EthanolConcentration',
        'FaceDetection',
        'FingerMovements',
        'HandMovementDirection',
        'Handwriting',
        'Heartbeat',
        'InsectWingbeat',
        'JapaneseVowels',
        'Libras',
        'LSST',
        'MotorImagery',
        'NATOPS',
        'PEMS-SF',
        'PenDigits',
        'Phoneme',
        'RacketSports',
        'SelfRegulationSCP1',
        'SelfRegulationSCP2',
        'SpokenArabicDigits',
        'StandWalkJump',
        'UWaveGestureLibrary'
    ]

    # datasets = ['LSST']
    # '--experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --selected_dataset LSST'
    #'--experiment_description exp1 --run_description run_1 --seed 123 --training_mode train-linear --selected_dataset LSST'

    experiment_description = 'exp4'
    run_description = f'run1'
    seed = 123
    num_gpu = 0

    for dataset in datasets:
        for training_mode in ['self_supervised', 'train-linear']:
            script_descriptor = open("main.py")
            a_script = script_descriptor.read()
            sys.argv = ["main.py", "--experiment_description", f"{experiment_description}", "--run_description", f"{run_description}",
                        "--seed", f"{seed}", "--training_mode", f"{training_mode}", "--selected_dataset", f"{dataset}", "--num_gpu", f"{num_gpu}"]
            exec(a_script)
            script_descriptor.close()
