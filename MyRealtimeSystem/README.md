# MyRealtimeSystem

## 概要
MyRealtimeSystemは、動画ファイルや画像フォルダを入力として、PyTorchモデルによるフレームごとの分類を行い、結果をCSVや画像として出力するリアルタイム推論システムです。

## 主な機能
- 動画または画像フォルダからのフレーム読み込み
- PyTorchモデルによるフレーム分類
- スライディングウィンドウによる平滑化
- 推論結果のCSV出力（確率・ラベル・平滑化後）
- タイムライン画像の生成

## ディレクトリ構成
- `main.cpp` : エントリーポイント
- `config.h` : 各種パスやパラメータの設定
- `src/` : 各種処理モジュール

### srcディレクトリ内モジュール説明

- `binarizer.h/cpp`  
  推論結果の確率値をしきい値処理し、ラベル（二値化）に変換するモジュール。

- `debug.h/cpp`  
  デバッグ用のユーティリティ関数やログ出力機能を提供。

- `predictor.h/cpp`  
  PyTorchモデルを用いた推論処理を担当。入力画像から特徴抽出・分類を行う。

- `result_writer.h/cpp`  
  推論結果（確率・ラベル・平滑化後ラベル）をCSVファイルとして出力する機能。

- `sliding_window.h/cpp`  
  スライディングウィンドウによる時系列データの平滑化処理を実装。

- `timeline_writer.h/cpp`  
  推論結果の時系列を可視化したタイムライン画像を生成・保存するモジュール。

- `video_loader.h/cpp`  
  動画ファイルや画像フォルダからフレームを読み込む機能。

## 必要環境
- C++17対応のコンパイラ
- OpenCV
- LibTorch（PyTorch C++ API）

## 使い方

1. 必要なライブラリ（OpenCV, LibTorchなど）をインストールしてください。
2. `config.h` で各種パスやパラメータを設定します。
3. Visual Studio 2022でプロジェクトをビルドします。
4. 実行ファイルを起動すると、指定した動画または画像フォルダに対して推論が行われ、結果が `outputs/` フォルダに出力されます。

## 設定例（config.h）
- `TREATMENT_MODEL_PATH` : PyTorchモデルのパス
- `VIDEO_PATH`           : 入力動画ファイルのパス
- `VIDEO_FOLDER_PATH`    : 入力画像フォルダのパス
- `OUTPUT_PROBS_CSV`     : 推論確率の出力先CSV
- `OUTPUT_LABELS_CSV`    : 推論ラベルの出力先CSV
- `OUTPUT_SMOOTHED_CSV`  : 平滑化後ラベルの出力先CSV
- `TIMELINE_IMAGE_PATH`  : タイムライン画像の出力先
