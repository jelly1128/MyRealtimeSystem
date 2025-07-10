#pragma once

// �V�[���A���u�A�����̊e�탉�x��ID��`
// ���탉�x���iscene�j
enum class OrganLabel {
	OUTSIDE_BODY = 0, // �̊O
	HEAD_NECK = 1,    // ����
	ESOPHAGUS = 2,    // �H��
	STOMACH = 3,      // ��
	DUODENUM = 4      // �\��w��
};

// ���u���x���itreatment scene�j
enum class TreatmentLabel {
    WHITE = 0,        // ���F��
	LUGOL = 1,        // ���S�[��
    INDIGO = 2,       // �C���f�B�S�J���~��
	NBI = 3,          // NBI�i���ш���j
	INSIDE_ROOM = 4,  // ����
	BUCKET = 5,       // �o�P�c
};

// ������q���x���ievent�̈��j
// BIOPSY = 1, ���̑� = 0 �ŏ���
constexpr int BIOPSY_PRESENT = 1;
constexpr int BIOPSY_ABSENT = 0;

