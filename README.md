## Project description
This fork contains the adaptation of the MotionAGFormer repository for the ACCAD [1] dataset

The project is done in Innopolis University as a final project in Data & Knowledge Representation course.

## Team members: 
- Karim Galliamov (k.galliamov@innopolis.university)
- Igor Abramov (ig.abramov@innopolis.university)
- Alexandra Voronova (a.voronova@innopolis.university)

For more details, please refer to the original repository and paper by S. Mehrabah, V. Adeli and B. Taati [2]

## Data
For conversion of raw ACCAD data to 3DHPE format please refer to https://github.com/goldbricklemon/AMASS-to-3DHPE

Afterwards, place the converted `accad.npz` file inside `data/raw` and execute ```python data/preprocess/accad.py```

## Reference
[1] https://accad.osu.edu/research/motion-lab/mocap-system-and-data

[2] https://github.com/taatiteam/motionagformer