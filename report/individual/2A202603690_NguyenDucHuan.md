# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603690
- **Họ và tên:** Nguyễn Đức Huân
- **SEED:** 174

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | ✅  |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅  |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | ✅  |

_.Kết quả độ trễ đo trên Colab CPU. Lần chạy trước trên GPU cho số khác (bị limit)._

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \

1.  BoW + LogReg (tự code) 0.974 0.755 0.851 0.824
2.  BoW + MLP (deep learning) 0.988 0.857 0.918 0.889  
    3a. Encoder (MiniLM) + kNN 0.832 0.959 0.891 0.976  
    3b. Encoder (MiniLM) + LogReg 0.989 0.929 0.958 0.947  
    3c. Encoder (MiniLM) + MLP head 0.978 0.908 0.942 0.919  
    4a. LLM nhỏ zero-shot 0.864 0.950 0.905 0.905  
    4b. LLM lớn zero-shot 1.000 0.950 0.974 0.974  
    4c. LLM nhỏ few-shot 1.000 0.900 0.947 0.947

                                     train (s)  tin/s (cả lô)  p50 ms/tin  \

3.  BoW + LogReg (tự code) 4.607 11937.788 0.059
4.  BoW + MLP (deep learning) 6.502 24714.219 0.287  
    3a. Encoder (MiniLM) + kNN 80.459 43.396 41.564  
    3b. Encoder (MiniLM) + LogReg 80.878 48.161 29.471  
    3c. Encoder (MiniLM) + MLP head 81.427 51.306 26.037  
    4a. LLM nhỏ zero-shot 0.000 1.881 359.500  
    4b. LLM lớn zero-shot 0.000 3.502 289.549  
    4c. LLM nhỏ few-shot 0.000 3.279 282.091

                                     p95 ms/tin  USD / 1k tin

5.  BoW + LogReg (tự code) 2.357 0.0
6.  BoW + MLP (deep learning) 0.431 0.0  
    3a. Encoder (MiniLM) + kNN 53.045 0.0  
    3b. Encoder (MiniLM) + LogReg 50.985 0.0  
    3c. Encoder (MiniLM) + MLP head 34.047 0.0  
    4a. LLM nhỏ zero-shot 1122.821 0.0  
    4b. LLM lớn zero-shot 356.892 0.0  
    4c. LLM nhỏ few-shot 410.219 0.0  
    • Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
    • SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
    • Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
    • train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

3 dự đoán lệch nhiều nhất là ml_lr, hard_eval và llm_models

1. ml_lr: lệch nhiều nhất, thứ tự gần như ngược lại

- Đoán: lr=0.001 cho 0.1, lr=50 cho 2. Dự đoán không ghi rõ đại lượng, nên mình hiểu đây là loss cuối.
- Kết quả: lr=0.001 cho loss cuối 0.626; lr=50 cho loss cuối 0.003, dù có lúc vọt lên khoảng 2.6 ở đầu.
  Vì sao lệch:
- lr=0.001: bước quá nhỏ nên sau 300 epoch loss mới giảm từ khoảng 0.693 xuống 0.626.
- lr=50: dự đoán giả định lr lớn sẽ khiến loss cuối cao. Thực tế bước đầu vượt quá xa, nhưng loss sau đó giảm mạnh.
  Cơ chế có thể là: loss của logistic regression là hàm lồi, và khi p đã sát nhãn thì error nhỏ lại nên bước cập nhật cũng nhỏ

2. hard_eval: đánh giá quá cao BoW và đoán sai nhóm khó nhất của LLM
   Đoán Kết quả
   BoW 75% 52–58%
   Embedding 80% 68–74%
   LLM 85% 88–90%
   Nhóm khó nhất với BoW natural_spam Đúng, BoW + LogReg và BoW + MLP cùng đạt 10%
   Nhóm khó nhất với LLM prompt_injection Sai, nhóm thấp nhất là “Tưởng spam nhưng không phải” (70%)

   BoW: các nhóm spam được viết để né từ khóa quen thuộc, nên BoW + LogReg chỉ đạt 10–17% ở 3 nhóm spam giọng tự nhiên, thao túng và né từ khóa. Ở 2 nhóm ham “Tưởng spam” và “Nói về spam”, nó vẫn đạt 100%, cho thấy mô hình nghiêng về đoán ham khi không thấy từ khóa spam. Điểm tổng vì thế tụt xuống gần mức đoán ngẫu nhiên.

3. llm_models: chênh lệch giữa hai model lớn hơn dự đoán
   Đoán: model lớn cao hơn nhưng “không hơn quá nhiều”, và LLM tốt hơn 3b.
   Kết quả trên SUB 40 tin:
   Model F1
   LLM nhỏ (gpt-oss-20b) 0.905
   3b (Encoder + LogReg) 0.947
   LLM lớn (gpt-oss-120b) 0.974

   Hai model LLM: chênh 0.069, vượt ngưỡng hòa 0.05. Model nhỏ sai thêm 3 tin ham viết kiểu nhắn tin thân mật hoặc thiếu ngữ cảnh, trong khi cả hai cùng sai tin spam về Arsenal. Vậy khoảng cách đến từ việc chặn nhầm ham, không phải bắt spam kém hơn

## 4. Phân tích 50 case khó

                                F1 test thường (SUB)  F1 50 case khó  \

1.  BoW + LogReg 0.824 0.333
2.  BoW + MLP 0.889 0.488  
    3a. Emb + kNN 0.976 0.764  
    3b. Emb + LogReg 0.947 0.680  
    3c. Emb + MLP head 0.919 0.680  
    4a. LLM nhỏ zero-shot 0.905 0.903  
    4d. LLM nhỏ + prompt phòng thủ 0.905 0.918

                                    acc 50 case khó

3.  BoW + LogReg 0.52
4.  BoW + MLP 0.58  
    3a. Emb + kNN 0.74  
    3b. Emb + LogReg 0.68  
    3c. Emb + MLP head 0.68  
    4a. LLM nhỏ zero-shot 0.88  
    4d. LLM nhỏ + prompt phòng thủ 0.90

BoW + LogReg tụt nhiều nhất: F1 giảm từ 0,824 xuống 0,333 (giảm 0,491), sau đó là BoW + MLP (0,889 xuống 0,488)

## 5. Thử thách cá nhân

🎯Đề #1: ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?
#1: ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1
p_te = predict_proba(X_te, w, b)
for thr in [0.2, 0.5, 0.8]:
pred = (p_te >= thr).astype(int)
m = metrics(y_te, pred)
fp = int(((pred == 1) & (y_te == 0)).sum())
fn = int(((pred == 0) & (y_te == 1)).sum())
print(f"ngưỡng {thr}: precision={m['precision']:.3f} · recall={m['recall']:.3f} · F1={m['f1']:.3f} · chặn nhầm={fp} · spam lọt={fn}")

**Kết quả:**
ngưỡng 0.2: precision=0.860 · recall=0.878 · F1=0.869 · chặn nhầm=14 · spam lọt=12
ngưỡng 0.5: precision=0.974 · recall=0.755 · F1=0.851 · chặn nhầm=2 · spam lọt=24
ngưỡng 0.8: precision=1.000 · recall=0.602 · F1=0.752 · chặn nhầm=0 · spam lọt=39

**Giải thích**

1. Tính xác suất mỗi email thuộc lớp Spam (lớp 1) dựa trên mô hình Logistic Regression và setup 3 ngưỡng khác nhau (0.2, 0.5, 0.8).
2. Gán nhãn dự đoán ( >= ngưỡng -> spam(1), ngược lại -> ham(0)).
3. Tính các chỉ số và ghi lại các trường hợp phân loại nhầm.

## 6. Phương án cho sản phẩm thật

Bốn cách đạt p95 < 50 ms là BoW + LogReg, BoW + MLP, 3c và 3b; 3a và mọi cách LLM đều không đạt. Tôi chọn 3b vì đạt cả độ trễ (p95 khoảng 47 ms) lẫn chất lượng (F1 test 0,958). Với 300 triệu tin/tháng, mỗi tin 100 token input và 1 token output, LLM tốn khoảng 3.150/tháng với gpt-6-luna và 31.500/tháng với gpt-6.1-terra. Dù vậy, p95 của LLM (371–755 ms) vẫn vượt xa yêu cầu.

## 7. AI log và điều đã học

| Phần                                        | Bạn hỏi AI gì (prompt)                                                       | AI trả lời/viết gì                                                                                                                                                                  | Bạn đã sửa / kiểm chứng gì                                                                |
| ------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Phần 1 · TODO 1.1                           | Hỏi cách làm; nhờ hướng dẫn vectorize                                        | Từ chối viết hoàn chỉnh; đưa dàn ý và cú pháp np.zeros, enumerate, kiểm tra khóa dict, gán ô ma trận                                                                                | Tự viết code từ gợi ý                                                                     |
| Phần 1 · TODO 1.3                           | Hỏi nguyên tắc gradient descent, ký hiệu và cách thực hiện                   | Giải thích 4 bước, ví dụ số, ký hiệu; gợi ý từng dòng bằng lời và cách kiểm tra lỗi                                                                                                 | [Xác nhận tự viết code, ô kiểm tra ✅]                                                    |
| Phần 1 · Thí nghiệm 1.B                     | Gửi biểu đồ kết quả của 3 giá trị lr                                         | Đính chính mô tả lr=50; giải thích đoạn tăng vọt rồi hội tụ, nhắc đây là loss train; đưa code so F1 train/test                                                                      | Biểu đồ cho thấy mô tả ban đầu chưa khớp. [Xác nhận kết quả so F1, nếu có chạy]           |
| Phần 2 · Khái niệm                          | Hỏi luồng hoạt động của MLP và khác biệt với mô hình tuyến tính              | Giải thích luồng dự đoán, luồng train, so sánh từng phần; đính chính ví dụ free/call và dùng XOR                                                                                    | [Xác nhận đã tự kiểm tra ví dụ XOR, nếu có]                                               |
| Phần 2 · TODO 2.2                           | Hỏi code train_mlp đúng chưa; nhờ giải thích gradient                        | Đọc code, không thấy lỗi nhưng chưa chạy; giải thích từng dòng, quy tắc chuỗi và backprop                                                                                           | 5 dòng TODO có trong code đã gửi. [Xác nhận ô kiểm tra ✅]                                |
| Phần 3 · TODO 3.1–3.2                       | Hỏi code cosine và top_k có đúng không; hỏi lỗi `NoneType - int`             | Chỉ ra hai hàm thiếu `return`, giải thích nguyên nhân lỗi; không viết sẵn lời giải                                                                                                  | Tự viết code và tự sửa theo góp ý. [Xác nhận ô kiểm tra ✅]                               |
| Phần 3 · Thí nghiệm 3.A                     | Hỏi cách làm thí nghiệm                                                      | Giải thích các bước và cách đọc kết quả; không đoán hộ                                                                                                                              | Tự viết dự đoán và chạy. [Nêu kết quả tự kiểm chứng]                                      |
| Phần 3 · TODO 3.3                           | Hỏi cách làm kNN                                                             | Nêu các bước và điểm dễ sai, dò tay ô test; không viết sẵn code                                                                                                                     | Tự viết `knn_predict`. [Xác nhận ô kiểm tra ✅]                                           |
| Phần 3 · Câu hỏi 3, Bắt lỗi 2, Checkpoint B | Nhờ soạn câu trả lời ngắn; gửi 2 case từ Thí nghiệm 3.B và bảng kết quả      | Soạn nháp câu trả lời, dùng số của tôi; suy ra TP/FP/FN từ bảng; đưa code xem trọng số từng từ                                                                                      | [Ghi phần đã viết lại, đã kiểm tra `y_te.sum()` và trọng số hay chưa]                     |
| Phần 4 · Cấu hình API                       | Hỏi cách lấy key và thêm vào Colab Secrets; hỏi OpenAI có API miễn phí không | Hướng dẫn thêm `GEMINI_API_KEY`, chọn `LLM_PROVIDER`; giải thích API của OpenAI thường phải trả phí                                                                                 | Tự tạo key và cấu hình; chạy ô cấu hình và thấy đúng provider                             |
| Phần 4 · Sửa lỗi khi chạy                   | Gửi lỗi 429 theo phút, `NameError`, lỗi 400 `MINIMAL`, lỗi 429 theo ngày     | Đề xuất thêm `time.sleep(4.5)`, chạy lại các ô sau khi runtime khởi động lại, đổi `reasoning_effort` sang `low`, chuyển sang Groq vì Gemini `gemini-3.8-flash` chỉ cho 20 lượt/ngày | Áp dụng các đề xuất; chuyển sang Groq và chạy xong 4.A (`invalid=0`) và 4.B (`invalid=0`) |
| Phần 5 · Câu hỏi 5                          | Gửi biểu đồ Pareto, nhờ phân tích lại câu trả lời                            | Đọc biểu đồ và soạn nháp; lưu ý 3a chỉ tốt hơn 3b trên SUB, độ trễ chỉ đọc gần đúng                                                                                                 | [Ghi phần đã thay số và kiểm tra]                                                         |
| Phần 5 · Stress test, Câu hỏi 6             | Hỏi ý nghĩa của phần dự đoán; gửi bảng kết quả                               | Giải thích câu hỏi; không đoán hộ; phân tích bảng và soạn nháp Câu hỏi 6                                                                                                            | Tự viết dự đoán và chạy stress test. [Ghi phần đã viết lại]                               |
| Phần 5 · Tính tay                           | Nhờ hướng dẫn; hỏi lại kết quả các câu                                       | Nêu công thức và các bước, để lại bước cuối; xác nhận 10,50 USD là đúng và trình bày phép tính                                                                                      | Tự tính trên giấy. [Xác nhận 5 dấu ✅ và ảnh `hand_calc.jpg`]                             |
