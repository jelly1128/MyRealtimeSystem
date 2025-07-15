# MyRealtimeSystem

## �T�v
MyRealtimeSystem�́A����t�@�C����摜�t�H���_����͂Ƃ��āAPyTorch���f���ɂ��t���[�����Ƃ̕��ނ��s���A���ʂ�CSV��摜�Ƃ��ďo�͂��郊�A���^�C�����_�V�X�e���ł��B

## ��ȋ@�\
- ����܂��͉摜�t�H���_����̃t���[���ǂݍ���
- PyTorch���f���ɂ��t���[������
- �X���C�f�B���O�E�B���h�E�ɂ�镽����
- ���_���ʂ�CSV�o�́i�m���E���x���E��������j
- �^�C�����C���摜�̐���

## �f�B���N�g���\��
- `main.cpp` : �G���g���[�|�C���g
- `config.h` : �e��p�X��p�����[�^�̐ݒ�
- `src/` : �e�폈�����W���[��

### src�f�B���N�g�������W���[������

- `binarizer.h/cpp`  
  ���_���ʂ̊m���l���������l�������A���x���i��l���j�ɕϊ����郂�W���[���B

- `debug.h/cpp`  
  �f�o�b�O�p�̃��[�e�B���e�B�֐��⃍�O�o�͋@�\��񋟁B

- `predictor.h/cpp`  
  PyTorch���f����p�������_������S���B���͉摜����������o�E���ނ��s���B

- `result_writer.h/cpp`  
  ���_���ʁi�m���E���x���E�������ド�x���j��CSV�t�@�C���Ƃ��ďo�͂���@�\�B

- `sliding_window.h/cpp`  
  �X���C�f�B���O�E�B���h�E�ɂ�鎞�n��f�[�^�̕����������������B

- `timeline_writer.h/cpp`  
  ���_���ʂ̎��n������������^�C�����C���摜�𐶐��E�ۑ����郂�W���[���B

- `video_loader.h/cpp`  
  ����t�@�C����摜�t�H���_����t���[����ǂݍ��ދ@�\�B

## �K�v��
- C++17�Ή��̃R���p�C��
- OpenCV
- LibTorch�iPyTorch C++ API�j

## �g����

1. �K�v�ȃ��C�u�����iOpenCV, LibTorch�Ȃǁj���C���X�g�[�����Ă��������B
2. `config.h` �Ŋe��p�X��p�����[�^��ݒ肵�܂��B
3. Visual Studio 2022�Ńv���W�F�N�g���r���h���܂��B
4. ���s�t�@�C�����N������ƁA�w�肵������܂��͉摜�t�H���_�ɑ΂��Đ��_���s���A���ʂ� `outputs/` �t�H���_�ɏo�͂���܂��B

## �ݒ��iconfig.h�j
- `TREATMENT_MODEL_PATH` : PyTorch���f���̃p�X
- `VIDEO_PATH`           : ���͓���t�@�C���̃p�X
- `VIDEO_FOLDER_PATH`    : ���͉摜�t�H���_�̃p�X
- `OUTPUT_PROBS_CSV`     : ���_�m���̏o�͐�CSV
- `OUTPUT_LABELS_CSV`    : ���_���x���̏o�͐�CSV
- `OUTPUT_SMOOTHED_CSV`  : �������ド�x���̏o�͐�CSV
- `TIMELINE_IMAGE_PATH`  : �^�C�����C���摜�̏o�͐�
