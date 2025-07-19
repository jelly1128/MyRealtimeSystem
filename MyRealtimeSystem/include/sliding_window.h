#pragma once
#include <vector>
#include <deque>
#include <optional>

/**
 * @brief �X���C�f�B���O�E�B���h�E�ŃV�[�����x���𑽐������o�i���_�͒��O���x���j
 *
 * @param treatmentLabels �S�N���X�i��F15�N���X�j�̃t���[�����Ƃ̃��x���z��i�e�v�f��one-hot/multi-hot�z��j
 * @param windowSize �X���C�f�B���O�E�B���h�E�̃T�C�Y
 * @param step �E�B���h�E�̈ړ��X�e�b�v
 * @param numMainClasses �V�[���N���X���i��F6�Ȃ�0�`5�N���X���Ώہj
 * @return �e�E�B���h�E�����t���[�����Ƃ̃V�[�����x���z��i�[�t���[���͏o�͂��Ȃ��j
 *
 * @details �e�E�B���h�E���ŃV�[���N���X���Ƃ̍��v�l���W�v���A�ő��N���X�����̃E�B���h�E�����t���[���̃��x���Ƃ���B
 *          �����N���X�������ő��̏ꍇ�͒��O�̏o�̓��x�����̗p�B�ŏ��̃E�B���h�E�̂ݍŏ��C���f�b�N�X�N���X�B
 */
std::vector<int> slidingWindowExtractSceneLabels(
    const std::vector<std::vector<int>>& treatmentLabels,
    int windowSize,
    int step,
    int numSceneClasses
);


std::optional<int> processSceneLabelSlidingWindow(
    const std::deque<std::vector<int>>& windowSceneLabelBuffer,
    int prevSceneLabel
);