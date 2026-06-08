# RAG Evaluation Results

## Framework sử dụng

- **Framework:** DeepEval
- **Judge model:** `gpt-4o-mini`
- **Threshold:** `0.7`
- **Golden dataset size:** `20`
- **Metrics:** Faithfulness, Answer Relevance, Context Recall, Context Precision

---

## Overall Scores

| Metric | Config A (hybrid + RRF rerank) | Config B (dense-only) | Δ |
|--------|-------------------------------:|----------------------:|---:|
| Faithfulness | 0.870 | 0.866 | 0.004 |
| Answer Relevance | 0.874 | 0.707 | 0.167 |
| Context Recall | 0.925 | 0.925 | 0.000 |
| Context Precision | 0.799 | 0.753 | 0.046 |
| **Average** | **0.867** | **0.813** | **0.054** |

---

## A/B Comparison Analysis

**Config A — hybrid + RRF rerank:**

Config A sử dụng retrieval pipeline đầy đủ: semantic search từ Weaviate, lexical search bằng BM25, merge bằng RRF và PageIndex fallback khi cần. Cấu hình này được kỳ vọng có context recall tốt hơn vì kết hợp cả dense retrieval và keyword matching.

**Config B — dense-only:**

Config B chỉ sử dụng semantic search từ Weaviate. Cấu hình này đơn giản hơn, nhưng có thể bỏ lỡ các câu hỏi cần keyword chính xác như số điều luật, tên tội danh, hoặc tên riêng trong bài báo.

**Kết luận:**

Config A tốt hơn hoặc tương đương Config B theo điểm trung bình. Điều này cho thấy hybrid retrieval giúp pipeline lấy evidence ổn định hơn, đặc biệt với dữ liệu pháp luật có nhiều thuật ngữ và số điều khoản.

---

## Worst Performers (Bottom 3 — Config A)

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|----------|-------------:|----------:|-------:|----------:|---------------|------------|
| 1 | Khi hỏi về một cá nhân trong bài báo, hệ thống RAG cần phân biệt những trạng thái pháp lý nào? | 0.714 | 0.286 | 0.000 | 0.000 | Retrieval | The score is 0.00 because the sentence discusses legal statuses that are not mentioned in the node(s) in retrieval context, indicating a complete lack of relevant information. |
| 2 | Nếu context chỉ nói một người bị khởi tố, hệ thống có nên trả lời rằng người đó bị kết án không? | 1.000 | 0.000 | 0.500 | 0.200 | Generation / Relevance | The score is 0.00 because the response fails to address the question about whether a person who has been prosecuted should be considered convicted, providing no relevant informa... |
| 3 | Tội chiếm đoạt chất ma túy được quy định tại điều nào? | 1.000 | 1.000 | 1.000 | 0.200 | Retrieval | The score is 0.20 because the relevant node, which is ranked 5th, is significantly outnumbered by the irrelevant nodes ranked 1st to 4th. The first four nodes discuss drug types... |

---

## Per-case Scores — Config A

| # | Question | Faithfulness | Relevance | Recall | Precision | Average |
|---|----------|-------------:|----------:|-------:|----------:|--------:|
| 1 | Hình phạt cơ bản cho tội tàng trữ trái phép chất ma túy theo Điều 249 Bộ luật Hình sự là gì? | 0.500 | 1.000 | 1.000 | 1.000 | 0.875 |
| 2 | Ngoài hình phạt tù, người phạm tội tàng trữ trái phép chất ma túy còn có thể chịu hình phạt bổ sung nào? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | Tội vận chuyển trái phép chất ma túy được quy định tại điều nào của Bộ luật Hình sự? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | Tội mua bán trái phép chất ma túy được quy định tại điều nào của Bộ luật Hình sự? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | Tội chiếm đoạt chất ma túy được quy định tại điều nào? | 1.000 | 1.000 | 1.000 | 0.200 | 0.800 |
| 6 | Luật Phòng, chống ma túy 2021 điều chỉnh những nhóm vấn đề chính nào? | 0.500 | 1.000 | 1.000 | 0.887 | 0.847 |
| 7 | Theo Luật Phòng, chống ma túy 2021, các hành vi nào bị nghiêm cấm liên quan đến chất ma túy? | 0.714 | 1.000 | 1.000 | 0.867 | 0.895 |
| 8 | Luật Phòng, chống ma túy 2021 quy định những hình thức cai nghiện ma túy nào? | 1.000 | 1.000 | 1.000 | 0.833 | 0.958 |
| 9 | Người sử dụng trái phép chất ma túy có thể bị quản lý như thế nào theo Luật Phòng, chống ma túy 2021? | 1.000 | 1.000 | 1.000 | 0.804 | 0.951 |
| 10 | Nghị định 105/2021/NĐ-CP hướng dẫn nội dung gì liên quan đến Luật Phòng, chống ma túy? | 0.714 | 1.000 | 1.000 | 1.000 | 0.929 |
| 11 | Nghị định 57/2022/NĐ-CP được dùng để tra cứu nội dung gì? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 12 | Nhóm I trong danh mục chất ma túy theo Nghị định 57/2022/NĐ-CP có ý nghĩa gì? | 0.500 | 1.000 | 1.000 | 1.000 | 0.875 |
| 13 | Theo các bài báo đã crawl, những người nổi tiếng nào được nhắc đến trong vụ việc liên quan đến ma túy cuối năm 2024? | 1.000 | 0.800 | 1.000 | 0.583 | 0.846 |
| 14 | Bài báo về chuyên án VN10 cho biết điều gì về Chi Dân và An Tây? | 0.750 | 1.000 | 1.000 | 1.000 | 0.938 |
| 15 | Các bài báo trong corpus nhắc đến diễn viên Hữu Tín liên quan đến hành vi nào? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16 | Nhà thiết kế Công Trí được nhắc đến trong corpus trong bối cảnh nào? | 1.000 | 0.400 | 1.000 | 0.806 | 0.801 |
| 17 | Các bài báo nhận định gì về trách nhiệm xã hội của nghệ sĩ khi vướng vào ma túy? | 1.000 | 1.000 | 1.000 | 0.804 | 0.951 |
| 18 | Vì sao corpus bài báo phù hợp để đánh giá RAG về chủ đề nghệ sĩ liên quan đến ma túy? | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 19 | Khi hỏi về một cá nhân trong bài báo, hệ thống RAG cần phân biệt những trạng thái pháp lý nào? | 0.714 | 0.286 | 0.000 | 0.000 | 0.250 |
| 20 | Nếu context chỉ nói một người bị khởi tố, hệ thống có nên trả lời rằng người đó bị kết án không? | 1.000 | 0.000 | 0.500 | 0.200 | 0.425 |

---

## Recommendations

### Cải tiến 1 — Làm sạch context từ bài báo
**Action:** Loại bỏ markdown ảnh, menu, navigation và footer khỏi các bài báo trước khi chunking.

**Expected impact:** Tăng Context Precision và giảm khả năng LLM cite nhầm đoạn nhiễu.

### Cải tiến 2 — Ưu tiên nguồn pháp luật cho câu hỏi pháp lý
**Action:** Thêm rule hoặc metadata filter: nếu query chứa `Điều`, `hình phạt`, `Bộ luật`, `Luật`, ưu tiên `type=legal`.

**Expected impact:** Tăng Faithfulness cho câu hỏi pháp luật, giảm nhiễu từ bài báo.

### Cải tiến 3 — Reranking mạnh hơn cho top candidates
**Action:** Thử cross-encoder reranker như Jina Reranker v2 cho top 20 candidates trước generation.

**Expected impact:** Cải thiện Context Precision và Answer Relevance, đặc biệt với câu hỏi có nhiều thực thể như nghệ sĩ/người nổi tiếng.

