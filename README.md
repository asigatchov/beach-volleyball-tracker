# 沙灘排球視覺分析系統 (Beach Volleyball Vision-Analytics)

本專案利用電腦視覺與深度學習技術，對沙灘排球比賽影片進行自動化分析，核心功能是從長時間錄影中分割出有效比賽片段，並基於球員姿態偵測發球等關鍵事件。

## 核心工作流程

本系統的分析流程分為三個主要階段：

**1. 場地幾何定義 (一次性設定)**
   - **目的**: 為特定比賽場地或攝影機角度定義必要的幾何資訊。
   - **腳本**: `python court_definition/court_config_generator.py <影片路徑>`
   - **產出**: 一個 `court_config.json` 檔案，包含：
     - `court_boundary_polygon`: 球場的四個邊界點。
     - `exclusion_zones`: 需要忽略的區域（如裁判席、觀眾席）。
     - `net_y`: 網子在畫面上的Y座標，用於簡單的位置判斷。
     - `background_ball_zones`: 需要過濾掉的靜態背景球區域。

**2. 影片自動分割 (基於分數變化)**
   - **目的**: 將原始長影片，根據分數板的變化，自動分割成多個獨立的比賽回合短影片。
   - **腳本**: `python video_processing/video_slicer_by_score.py --input <長影片路徑> --output_dir <輸出目錄>`
   - **方法**: 透過比較分數板ROI的影像差異 (SAD - Sum of Absolute Differences) 來偵測變化，而非OCR，這更穩健。
   - **產出**: 在輸出目錄下生成 `normal_segments/` 和 `long_segments/`，存放分割好的影片。

**3. 追蹤與事件分析 (核心)**
   - **目的**: 對分割後的短影片進行逐幀分析，偵測球與球員，提取姿態，並最終識別出發球事件。
   - **腳本**: `python video_processing/track_ball_and_player.py --input <單個短影片路徑>`
   - **方法**:
     1. **讀取設定**: 載入 `court_config.json`。
     2. **物件偵測**: 使用YOLO模型偵測球 (`ball_best.pt`) 和球員 (`yolov8s-pose.pt`)。
     3. **球員篩選**: 透過「排除區 -> 場內優先 -> 距離中心」的策略篩選出4名主要球員。
     4. **姿態提取**: 儲存篩選後球員的17個身體關鍵點。
     5. **數據儲存**: 將每一幀的所有資訊（球、球員、姿態）寫入一個詳細的 `..._all_frames_data_with_pose.json` 檔案。
   - **後續 (自動調用)**:
     - 在追蹤完成後，可以接著調用 `event_analyzer.py` 中的 `find_serve_events` 函數。
     - 該函數會讀取剛剛生成的JSON檔案，**基於手腕速度、手腕與肩膀相對位置、手與球的接近程度** 來識別發球擊球點。

## 環境設定

1.  **安裝依賴**:
    ```bash
    pip install -r requirements.txt
    ```
    主要依賴包括 `ultralytics`, `opencv-python`, `numpy`。

2.  **準備模型**:
    - 將您的 `ball_best.pt` 模型放入 `models/` 資料夾。
    - 下載或準備一個YOLOv8姿態模型，如 `yolov8s-pose.pt`，也放入 `models/` 資料夾。

## 如何使用

1.  **定義場地 (首次)**:
    ```bash
    python court_definition/court_config_generator.py "path/to/your/sample_video.mp4"
    ```
    按照螢幕提示，用滑鼠點擊定義場地、排除區和網子位置。這會生成 `court_config.json`。

2.  **分割影片**:
    ```bash
    python video_processing/video_slicer_by_score.py --input "path/to/long_match.mp4" --output_dir "output_data/match1_segments"
    ```

3.  **分析單個片段**:
    對 `output_data/match1_segments/normal_segments` 中的某個影片執行分析。
    ```bash
    python video_processing/track_ball_and_player.py --input "output_data/match1_segments/normal_segments/segment_001.mp4"
    ```
    腳本執行完畢後，會在對應的輸出檔案夾內生成標註好的影片和包含所有姿態數據的JSON檔案。您可以進一步整合事件分析的邏輯，或手動檢查JSON結果。

## 未來展望

* **自動化流程串接**: 將分割和分析腳本串聯起來，實現全自動處理。
* **球員ID追蹤**: 引入追蹤演算法（如 DeepSORT, BoT-SORT）為每個球員分配一個穩定的ID，以應對遮擋。
* **更複雜事件偵測**: 基於姿態數據，開發扣球、攔網等更複雜事件的識別模型。
* **軌跡與戰術分析**: 擬合球的飛行軌跡，進行落點預測和戰術繪圖。