#pragma once
#include <vector>

// �e�t���[���ɂ�����ŏI�I�ȃ��x�����i�o�͗p�j
struct FrameLabel {
    int frameIndex;             // �t���[���ԍ�
    int organLabel;             // ���탉�x���iint�^�BOrganLabel�ɏ����j
    int treatmentSceneLabel;    // ���u�V�[�����x���iint�^�j
    int biopsyLabel;            // �������x���i0 or 1�j
};

// �T���l�C���I���X�R�A�����p�̃X�R�A�t���\���́i�������ԗp�j
struct FrameData {
    int frameIndex;                     // �t���[���ԍ�
    std::vector<float> probabilities;   // ���_�X�R�A�i15�N���X�j
    int swLabel;                        // ���������ꂽ���x��
    float S_target = 0.0f;              // �^�[�Q�b�g�X�R�A
    float S_event = 0.0f;               // �C�x���g�X�R�A
};