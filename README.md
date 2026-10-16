# Dự báo doanh số và quản trị tồn kho

Ứng dụng Streamlit hỗ trợ phân tích doanh số, tạo đặc trưng chuỗi thời gian,
huấn luyện LSTM/GRU, lập dự báo và ước tính nhu cầu bổ sung hàng. Các chức năng
được chia thành trang độc lập trong thư mục `app_pages/`.

## Chạy ứng dụng

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Các trang

- **Mô hình huấn luyện** — huấn luyện LSTM/GRU (Deep Learning) và xem kết quả
  kiểm định trên dữ liệu đang mở.
- **Dự báo doanh số** — tạo dự báo từ mô hình đã huấn luyện và tải CSV.
- **Tạo đặc trưng dự báo** — tạo, xem và tải thông tin lịch, khuyến mãi, độ trễ
  và trung bình trượt để phục vụ phân tích, dự báo.
- **Tiền xử lý dữ liệu** — kiểm tra dữ liệu thiếu/trùng/outlier và làm sạch dữ
  liệu nguồn.
- **Phân tích dữ liệu (EDA)** — cơ cấu doanh số, bản đồ và hiệu suất cửa hàng, thư
  viện kiến thức, trợ lý tra cứu, khuyến mãi, ngày lễ, giá dầu và outlier. Bản
  đồ hiển thị khi dữ liệu có thành phố khớp với danh mục vị trí Ecuador; trợ lý
  tra cứu dựa trên `knowledge_base.json`.
- **Phân tích xu hướng bán hàng** — xu hướng theo ngày/tuần/tháng, mùa vụ theo
  thứ và nhóm hàng nổi bật.
- **Đánh giá mô hình** — benchmark Favorita có sẵn, backtest và thử nghiệm loại
  bỏ từng nguồn dữ liệu.
- **Dashboard** — tổng quan doanh số, trạng thái mô hình, dự báo và tồn kho.
- **Tồn kho an toàn & Đặt hàng lại** — mức dự trữ an toàn, điểm cần đặt hàng
  lại, lượng đặt tối ưu, mô phỏng tồn kho và phân loại ABC.
- **Cảnh báo tồn kho** — đối chiếu tồn kho người dùng nhập với nhu cầu 28 ngày
  gần nhất theo nhóm hàng.
- **Dữ liệu bán hàng** — nhập CSV/Excel/ZIP, kết nối Supabase hoặc tải dữ liệu
  mẫu theo ERD.
- **Báo cáo tổng hợp** — xem và tải dự báo, phân loại ABC và dữ liệu hiện hành.

Benchmark trên trang **Đánh giá mô hình** là kết quả cố định từ bộ Favorita;
trang **Mô hình huấn luyện** huấn luyện mô hình trên dữ liệu trong phiên hiện tại.
Tồn kho thực tế không nằm trong ERD, vì vậy các trang tồn kho cần người dùng
nhập số lượng hiện có.

## Định dạng dữ liệu

Tải từng bảng CSV hoặc ZIP chứa một phần/toàn bộ các bảng sau. Tên tệp nên dùng
tên bảng để tự động nhận diện:

| Bảng | Cột chính | Khóa / liên kết |
|---|---|---|
| `train2.csv` (`train.csv`) | `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion` | `id`; `store_nbr` → stores |
| `test.csv` | `id`, `date`, `store_nbr`, `family`, `onpromotion` | `id`; khóa dự báo tương ứng với sample_submission |
| `stores.csv` | `store_nbr`, `city`, `state`, `type`, `cluster` | `store_nbr` |
| `transactions.csv` | `date`, `store_nbr`, `transactions` | `date + store_nbr` |
| `oil.csv` | `date`, `dcoilwtico` | `date` |
| `holidays_events.csv` | `date`, `type`, `locale`, `locale_name`, `description`, `transferred` | ngày có thể có nhiều sự kiện |
| `sample_submission.csv` | `id`, `sales` | `id` |

Các bảng phụ được nối theo khóa ở trên; sự kiện ngày lễ được gộp theo ngày trước
khi nối để tránh nhân bản dòng doanh số. Dữ liệu tổng quát cũng được chấp nhận
nếu có các cột ngày, nhóm hàng/sản phẩm và doanh số/số lượng (ví dụ:
`date, product, quantity`).

Tồn kho thực tế không nằm trong ERD; ứng dụng yêu cầu nhập tồn kho hiện tại và
lead time để tính ROP. Dữ liệu mẫu tạo một bộ nhỏ có cửa hàng, nhóm hàng,
khuyến mãi, giao dịch, giá dầu và ngày lễ.

## Kết nối Supabase (có thể bật khi sẵn sàng)

Trang **Dữ liệu → Supabase** đã hỗ trợ đọc các bảng theo ERD. Để bật kết nối,
đặt thông tin dự án trong môi trường máy chủ Streamlit hoặc secrets; không ghi
khóa vào mã nguồn hoặc Git:

```text
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<key-with-appropriate-RLS-policies>
```

Có thể dùng `.streamlit/secrets.toml` cục bộ (đưa file này vào `.gitignore`):

```toml
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_KEY = "<key-with-appropriate-RLS-policies>"
```

Để lưu dữ liệu tải lên và khôi phục sau khi mở lại ứng dụng, tạo thêm hai bảng
snapshot trong Supabase SQL Editor:

```sql
create table if not exists public.app_dataset_snapshots (
    id uuid primary key,
    created_at timestamptz not null default now(),
    source_label text not null,
    table_metadata jsonb not null,
    is_complete boolean not null default false
);

create table if not exists public.app_dataset_snapshot_rows (
    snapshot_id uuid not null references public.app_dataset_snapshots(id) on delete cascade,
    table_name text not null,
    row_index bigint not null,
    row_data jsonb not null,
    primary key (snapshot_id, table_name, row_index)
);

create index if not exists app_dataset_snapshots_latest_idx
    on public.app_dataset_snapshots (created_at desc)
    where is_complete = true;

alter table public.app_dataset_snapshots enable row level security;
alter table public.app_dataset_snapshot_rows enable row level security;
grant select, insert, update on public.app_dataset_snapshots to service_role;
grant select, insert on public.app_dataset_snapshot_rows to service_role;
```

Trong cấu hình máy chủ Streamlit, đặt `SUPABASE_KEY` thành secret/service-role
key có quyền truy cập các bảng snapshot; không dùng key này trong trình duyệt,
không ghi vào repository và không chia sẻ công khai. Sau khi nạp file hoặc tạo
dữ liệu mẫu, ứng dụng tự lưu một snapshot mới mà không xóa các snapshot cũ.
Khi khởi tạo phiên mới, ứng dụng tự khôi phục snapshot hoàn chỉnh mới nhất;
trang **Dữ liệu → Supabase** cũng cho phép chọn và khôi phục snapshot cũ.

Sau khi cài dependencies từ `requirements.txt`, ứng dụng đọc dữ liệu ở phía
máy chủ và giới hạn số dòng gần nhất của `train2`/`test`. Dùng nút **Nạp từ
Supabase** trong trang Dữ liệu để tải dữ liệu vào phiên làm việc.

Đảm bảo các bảng `train2`, `test`, `stores`, `transactions`, `oil`,
`holidays_events`, `sample_submission` tồn tại với quyền `SELECT` cần thiết.
Với các bảng nguồn hiện có, có thể dùng khóa anon/publishable kèm chính sách
RLS chỉ đọc phù hợp. Tính năng snapshot cần quyền ghi; chỉ dùng secret/service-role
key trong môi trường máy chủ tin cậy và tuyệt đối không đưa khóa đó vào trình
duyệt hoặc repository.

Snapshot chỉ lưu các bảng dữ liệu nguồn đã nạp; trạng thái mô hình đã huấn luyện,
kết quả dự báo và dữ liệu tiền xử lý không được snapshot.
