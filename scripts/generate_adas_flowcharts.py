"""Sinh bộ flowchart draw.io cho kiến trúc ADAS mini.

Mục tiêu của bộ sơ đồ này là đọc được nhanh trong diagrams.net:

* mỗi khối mô tả một nhiệm vụ, không ghi chi tiết câu lệnh/hàm,
* dây đi theo luồng trái -> phải hoặc trên -> dưới,
* không dùng dây vòng dài; các bước lặp được biểu diễn bằng khối nhiệm vụ,
* các trang bám theo hướng multi-threading của runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from xml.sax.saxutils import escape


FLOWCHART_DIR = Path("docs/flowcharts")
OUT_PATH = FLOWCHART_DIR / "adas_project_flowcharts.drawio"
PAGE_W = 1700
PAGE_H = 1200
PAGE_MARGIN_X = 180
PAGE_MARGIN_Y = 180


@dataclass(frozen=True)
class Node:
    id: str
    text: str
    x: int
    y: int
    w: int
    h: int
    kind: str = "process"


@dataclass(frozen=True)
class Edge:
    id: str
    src: str
    dst: str
    label: str = ""
    color: str = "#4D4D4D"
    points: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Page:
    id: str
    name: str
    title: str
    filename: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    page_w: int = PAGE_W
    page_h: int = PAGE_H


STYLES = {
    "title": (
        "text;html=1;strokeColor=none;fillColor=none;fontSize=28;fontStyle=1;"
        "fontColor=#000000;align=center;verticalAlign=middle;"
    ),
    "terminator": (
        "shape=mxgraph.flowchart.terminator;whiteSpace=wrap;html=1;"
        "fillColor=#D5E8D4;strokeColor=#82B366;fontSize=15;fontStyle=1;fontColor=#000000;"
        "align=center;verticalAlign=middle;spacing=10;strokeWidth=1.5;"
    ),
    "data": (
        "shape=mxgraph.flowchart.data;whiteSpace=wrap;html=1;"
        "fillColor=#DAE8FC;strokeColor=#6C8EBF;fontSize=15;fontColor=#000000;"
        "align=center;verticalAlign=middle;spacing=10;strokeWidth=1.5;"
    ),
    "process": (
        "shape=mxgraph.flowchart.process;whiteSpace=wrap;html=1;"
        "fillColor=#FFF2CC;strokeColor=#D6B656;fontSize=15;fontColor=#000000;"
        "align=center;verticalAlign=middle;spacing=10;strokeWidth=1.5;"
    ),
    "predef": (
        "shape=mxgraph.flowchart.predefined_process;whiteSpace=wrap;html=1;"
        "fillColor=#E1D5E7;strokeColor=#9673A6;fontSize=15;fontColor=#000000;"
        "align=center;verticalAlign=middle;spacing=10;strokeWidth=1.5;"
    ),
    "decision": (
        "shape=mxgraph.flowchart.decision;whiteSpace=wrap;html=1;"
        "fillColor=#F8CECC;strokeColor=#B85450;fontSize=15;fontStyle=1;fontColor=#000000;"
        "align=center;verticalAlign=middle;spacing=10;strokeWidth=1.5;"
    ),
    "document": (
        "shape=mxgraph.flowchart.document;whiteSpace=wrap;html=1;"
        "fillColor=#F5F5F5;strokeColor=#666666;fontSize=13;fontColor=#000000;"
        "align=center;verticalAlign=middle;spacing=10;strokeWidth=1.2;"
    ),
    "note": (
        "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;"
        "fillColor=#F5F5F5;strokeColor=#666666;fontSize=13;fontColor=#000000;"
        "align=center;verticalAlign=middle;spacing=10;strokeWidth=1.2;"
    ),
    "header": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        "fillColor=#E6F2FF;strokeColor=#6C8EBF;fontSize=14;fontStyle=1;fontColor=#000000;"
        "align=center;verticalAlign=middle;spacing=8;strokeWidth=1.2;"
    ),
}


def N(id_: str, text: str, x: int, y: int, w: int, h: int, kind: str = "process") -> Node:
    return Node(id_, text, x, y, w, h, kind)


def E(
    id_: str,
    src: str,
    dst: str,
    label: str | tuple[tuple[int, int], ...] = "",
    color: str | tuple[tuple[int, int], ...] = "#4D4D4D",
    points: tuple[tuple[int, int], ...] = (),
) -> Edge:
    if not isinstance(label, str):
        points = tuple(label)
        label = ""
    if not isinstance(color, str):
        points = tuple(color)
        color = "#4D4D4D"
    return Edge(id_, src, dst, label, color, tuple(points))


def xml_text(text: str) -> str:
    return escape(text).replace("\n", "&lt;br&gt;")


def node_xml(node: Node) -> str:
    return (
        f'<mxCell id="{node.id}" value="{xml_text(node.text)}" '
        f'style="{STYLES[node.kind]}" vertex="1" parent="1">'
        f'<mxGeometry x="{node.x}" y="{node.y}" width="{node.w}" height="{node.h}" as="geometry"/>'
        "</mxCell>"
    )


def edge_xml(edge: Edge) -> str:
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        f"html=1;endArrow=block;strokeWidth=2;strokeColor={edge.color};"
        "fontSize=13;fontColor=#000000;labelBackgroundColor=#FFFFFF;"
    )
    if edge.points:
        points_xml = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in edge.points)
        geometry = f'<mxGeometry relative="1" as="geometry"><Array as="points">{points_xml}</Array></mxGeometry>'
    else:
        geometry = '<mxGeometry relative="1" as="geometry"/>'
    return (
        f'<mxCell id="{edge.id}" value="{xml_text(edge.label)}" style="{style}" '
        f'edge="1" parent="1" source="{edge.src}" target="{edge.dst}">'
        f"{geometry}</mxCell>"
    )


def resolved_page_size(page: Page) -> tuple[int, int]:
    max_x = page.page_w
    max_y = page.page_h
    for node in page.nodes:
        max_x = max(max_x, node.x + node.w)
        max_y = max(max_y, node.y + node.h)
    for edge in page.edges:
        for x, y in edge.points:
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    return max(page.page_w, max_x + PAGE_MARGIN_X), max(page.page_h, max_y + PAGE_MARGIN_Y)


def title_xml(page: Page, page_w: int) -> str:
    width = min(1120, page_w - 120)
    x = (page_w - width) // 2
    return (
        f'<mxCell id="{page.id}-title" value="{xml_text(page.title)}" '
        f'style="{STYLES["title"]}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="25" width="{width}" height="50" as="geometry"/>'
        "</mxCell>"
    )


def page_xml(page: Page) -> str:
    page_w, page_h = resolved_page_size(page)
    body = [title_xml(page, page_w)]
    body.extend(node_xml(node) for node in page.nodes)
    body.extend(edge_xml(edge) for edge in page.edges)
    return f"""
  <diagram id="{page.id}" name="{xml_text(page.name)}">
    <mxGraphModel dx="{page_w}" dy="{min(page_h, 1200)}" grid="1" gridSize="10" guides="1" tooltips="1"
      connect="1" arrows="1" fold="1" page="1" pageScale="1"
      pageWidth="{page_w}" pageHeight="{page_h}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {''.join(body)}
      </root>
    </mxGraphModel>
  </diagram>"""


def mxfile_xml(diagrams: tuple[str, ...], modified: str) -> str:
    return (
        f'<mxfile host="app.diagrams.net" modified="{modified}" '
        'agent="Codex" version="24.7.17" type="device">'
        + "".join(diagrams)
        + "\n</mxfile>\n"
    )


def existing_modified(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r'\bmodified="([^"]+)"', text)
    return match.group(1) if match else None


def write_mxfile_if_changed(path: Path, diagrams: tuple[str, ...], modified: str) -> bool:
    previous_modified = existing_modified(path)
    if previous_modified is not None and path.exists():
        previous_text = path.read_text(encoding="utf-8")
        unchanged_candidate = mxfile_xml(diagrams, previous_modified)
        if unchanged_candidate == previous_text:
            return False
    path.write_text(mxfile_xml(diagrams, modified), encoding="utf-8")
    return True


GENERATED_PAGES: tuple[Page, ...] = (
    Page(
        "p01-system",
        "01 - Tổng thể ADAS mini",
        "1. Tổng thể ADAS mini: các luồng chính và dữ liệu ra/vào",
        "01_tong_the_adas_mini.drawio",
        (
            N("s-start", "Khởi động hệ thống", 700, 90, 260, 55, "terminator"),
            N("s-config", "Đọc cấu hình chạy", 70, 210, 250, 80, "data"),
            N("s-source", "Mở nguồn ảnh", 70, 340, 250, 80, "predef"),
            N("s-backend", "Chuẩn bị bộ suy luận", 70, 470, 250, 80, "predef"),
            N("s-capture", "Luồng capture\nlấy frame mới", 400, 340, 250, 90, "predef"),
            N("s-fanout", "Sao chép frame\ncho các nhánh xử lý", 400, 500, 250, 95, "data"),
            N("s-lq", "Hàng đợi lane\nframe mới nhất", 730, 260, 240, 85, "data"),
            N("s-oq", "Hàng đợi object\nframe mới nhất", 730, 580, 240, 85, "data"),
            N("s-lane", "Luồng lane\nnhận diện làn đường", 1040, 240, 250, 105, "predef"),
            N("s-object", "Luồng object\nnhận diện vật thể", 1040, 560, 250, 105, "predef"),
            N("s-lane-out", "Kết quả lane", 1360, 260, 230, 80, "data"),
            N("s-object-out", "Kết quả object", 1360, 580, 230, 80, "data"),
            N("s-fusion", "Luồng fusion\nghép kết quả cùng frame", 1360, 410, 230, 105, "predef"),
            N("s-behavior", "Quyết định điều khiển", 1360, 740, 230, 85),
            N("s-output", "Luồng output\nan toàn, hiển thị, RC", 1040, 740, 250, 105, "predef"),
            N("s-stop", "Dừng hệ thống\nđóng tài nguyên", 700, 740, 260, 85, "terminator"),
            N(
                "s-note",
                "Cách đọc: dữ liệu đi từ trái sang phải. Hai nhánh lane và object chạy song song rồi gặp lại ở fusion.",
                70,
                720,
                480,
                115,
                "document",
            ),
        ),
        (
            E("s-e1", "s-start", "s-config", points=((830, 165), (195, 165), (195, 210))),
            E("s-e2", "s-config", "s-source"),
            E("s-e3", "s-source", "s-backend"),
            E("s-e4", "s-backend", "s-capture", "sẵn sàng", points=((320, 510), (360, 510), (360, 385), (400, 385))),
            E("s-e5", "s-capture", "s-fanout", "frame"),
            E("s-e6", "s-fanout", "s-lq", "vào lane", points=((650, 545), (690, 545), (690, 305), (730, 305))),
            E("s-e7", "s-fanout", "s-oq", "vào object", points=((650, 545), (690, 545), (690, 625), (730, 625))),
            E("s-e8", "s-lq", "s-lane"),
            E("s-e9", "s-oq", "s-object"),
            E("s-e10", "s-lane", "s-lane-out", "ra lane"),
            E("s-e11", "s-object", "s-object-out", "ra object"),
            E("s-e12", "s-lane-out", "s-fusion", points=((1475, 340), (1475, 410))),
            E("s-e13", "s-object-out", "s-fusion", points=((1475, 580), (1475, 515))),
            E("s-e14", "s-fusion", "s-behavior", "quyết định"),
            E("s-e15", "s-behavior", "s-output", "lệnh an toàn"),
            E("s-e16", "s-output", "s-stop", "khi dừng"),
        ),
        page_h=900,
    ),
    Page(
        "p02-frame",
        "02 - Luồng xử lý một frame",
        "2. Một frame đi qua các hàng đợi và worker",
        "02_luong_xu_ly_moi_frame.drawio",
        (
            N("f-start", "Frame mới", 705, 90, 190, 55, "terminator"),
            N("f-capture", "Nhận frame từ camera/video", 80, 220, 310, 85, "predef"),
            N("f-valid", "Frame dùng được?", 140, 355, 190, 105, "decision"),
            N("f-packet", "Tạo gói frame\nid, thời điểm, kích thước", 80, 520, 310, 95, "data"),
            N("f-fanout", "Phân phối frame\ncho từng nhánh đang bật", 80, 680, 310, 95, "data"),
            N("f-lane-q", "Vào hàng đợi lane", 510, 420, 270, 75, "data"),
            N("f-lane-work", "Xử lý làn đường", 510, 545, 270, 90, "predef"),
            N("f-lane-out", "Ra kết quả lane", 510, 690, 270, 75, "data"),
            N("f-obj-q", "Vào hàng đợi object", 510, 850, 270, 75, "data"),
            N("f-obj-work", "Xử lý vật thể", 510, 975, 270, 90, "predef"),
            N("f-obj-out", "Ra kết quả object", 510, 1120, 270, 75, "data"),
            N("f-sync", "Đồng bộ kết quả\ntheo frame", 930, 620, 290, 100, "predef"),
            N("f-ready", "Đủ dữ liệu cần thiết?", 980, 780, 190, 105, "decision"),
            N("f-stale", "Loại kết quả quá cũ", 1270, 800, 250, 80),
            N("f-fuse", "Tạo quyết định tổng hợp", 930, 955, 290, 90, "predef"),
            N("f-output-q", "Đưa sang output", 930, 1100, 290, 75, "data"),
            N("f-output", "Hiển thị hoặc gửi điều khiển", 930, 1225, 290, 90, "predef"),
            N("f-end", "Frame hoàn tất", 980, 1370, 190, 55, "terminator"),
            N(
                "f-note",
                "Các nhánh song song không lấy chung một hàng đợi. Mỗi worker nhận bản frame mới nhất của riêng nó.",
                1240,
                1090,
                330,
                120,
                "document",
            ),
        ),
        (
            E("f-e1", "f-start", "f-capture", points=((800, 165), (235, 165), (235, 220))),
            E("f-e2", "f-capture", "f-valid"),
            E("f-e3", "f-valid", "f-packet", "có", "#82B366"),
            E("f-e4", "f-valid", "f-end", "không", "#B85450", ((140, 408), (40, 408), (40, 1398), (980, 1398))),
            E("f-e5", "f-packet", "f-fanout"),
            E("f-e6", "f-fanout", "f-lane-q", "lane", ((390, 728), (450, 728), (450, 458), (510, 458))),
            E("f-e7", "f-lane-q", "f-lane-work"),
            E("f-e8", "f-lane-work", "f-lane-out", "lane ra"),
            E("f-e9", "f-fanout", "f-obj-q", "object", ((390, 728), (450, 728), (450, 888), (510, 888))),
            E("f-e10", "f-obj-q", "f-obj-work"),
            E("f-e11", "f-obj-work", "f-obj-out", "object ra"),
            E("f-e12", "f-lane-out", "f-sync", points=((780, 728), (880, 728), (880, 655), (930, 655))),
            E("f-e13", "f-obj-out", "f-sync", points=((780, 1158), (880, 1158), (880, 685), (930, 685))),
            E("f-e14", "f-sync", "f-ready"),
            E("f-e15", "f-ready", "f-fuse", "có", "#82B366"),
            E("f-e16", "f-ready", "f-stale", "chưa", "#D6B656"),
            E("f-e17", "f-stale", "f-fuse", "tiếp tục", ((1395, 880), (1395, 1000), (1220, 1000))),
            E("f-e18", "f-fuse", "f-output-q"),
            E("f-e19", "f-output-q", "f-output"),
            E("f-e20", "f-output", "f-end"),
        ),
        page_h=1460,
    ),
    Page(
        "p03-lane",
        "03 - Lane worker",
        "3. Lane worker: từ frame đến kết quả làn đường",
        "03_nhanh_phat_hien_lan_duong.drawio",
        (
            N("l-start", "Bắt đầu lane worker", 690, 90, 230, 55, "terminator"),
            N("l-in", "Nhận frame từ hàng đợi lane", 630, 185, 350, 80, "data"),
            N("l-pre", "Chuẩn bị ảnh cho mô hình lane", 630, 320, 350, 80),
            N("l-model", "Chạy mô hình phân đoạn làn", 630, 455, 350, 85, "predef"),
            N("l-decode", "Giải mã mask và vùng làn", 630, 595, 350, 90),
            N("l-area", "Lấy đường giữa từ vùng làn", 180, 755, 330, 85),
            N("l-boundary", "Theo dõi biên làn màu tím", 630, 750, 350, 95, "predef"),
            N("l-red", "Tạo đường tâm đỏ", 1070, 755, 330, 85, "predef"),
            N("l-path", "Chọn đường đi ổn định", 630, 920, 350, 90),
            N("l-corridor", "Tạo hành lang di chuyển", 630, 1065, 350, 90, "predef"),
            N("l-quality", "Tính độ tin cậy lane", 630, 1210, 350, 85),
            N("l-out", "Xuất gói kết quả lane", 630, 1350, 350, 90, "data"),
            N("l-end", "Đưa vào hàng đợi fusion", 685, 1500, 240, 55, "terminator"),
            N(
                "l-note",
                "State của tracker nằm trong lane worker. Các luồng sau chỉ nhận gói kết quả, không đọc trực tiếp state nội bộ.",
                80,
                1315,
                430,
                120,
                "document",
            ),
        ),
        (
            E("l-e1", "l-start", "l-in"),
            E("l-e2", "l-in", "l-pre"),
            E("l-e3", "l-pre", "l-model"),
            E("l-e4", "l-model", "l-decode"),
            E("l-e5", "l-decode", "l-area", "vùng làn", ((630, 640), (345, 640), (345, 755))),
            E("l-e6", "l-decode", "l-boundary", "biên làn"),
            E("l-e7", "l-boundary", "l-red", "biên ổn định"),
            E("l-e8", "l-area", "l-path", "fallback", ((345, 840), (345, 965), (630, 965))),
            E("l-e9", "l-red", "l-path", "ưu tiên", ((1235, 840), (1235, 965), (980, 965))),
            E("l-e10", "l-path", "l-corridor"),
            E("l-e11", "l-corridor", "l-quality"),
            E("l-e12", "l-quality", "l-out"),
            E("l-e13", "l-out", "l-end"),
        ),
        page_h=1580,
    ),
    Page(
        "p04-corridor",
        "04 - Lane/corridor geometry",
        "4. Geometry lane/corridor: biến lane thành hành lang xe chạy",
        "04_bev_ipm.drawio",
        (
            N("b-start", "Bắt đầu dựng corridor", 685, 90, 240, 55, "terminator"),
            N("b-in", "Đầu vào lane\nmask, biên, đường tâm", 620, 185, 370, 95, "data"),
            N("b-candidate", "Tìm ứng viên biên làn", 620, 335, 370, 85, "predef"),
            N("b-width", "Ước lượng bề rộng làn", 620, 475, 370, 85),
            N("b-check", "Biên làn tin cậy?", 705, 620, 200, 110, "decision"),
            N("b-fixed", "Dùng corridor mặc định", 190, 800, 330, 85, "data"),
            N("b-purple", "Giữ các biên phát hiện được", 620, 800, 370, 85),
            N("b-red", "Căn đường tâm theo vị trí xe", 620, 940, 370, 85, "predef"),
            N("b-smooth", "Làm mượt độ cong", 620, 1080, 370, 85),
            N("b-yellow", "Tạo corridor vàng", 620, 1220, 370, 85, "predef"),
            N("b-out", "Xuất guidance\ncho localizer và overlay", 620, 1360, 370, 90, "data"),
            N("b-end", "Corridor sẵn sàng", 685, 1510, 240, 55, "terminator"),
            N(
                "b-note",
                "Trang này mô tả geometry đang dùng trong runtime, không phải luồng BEV/IPM riêng.",
                1080,
                1060,
                370,
                105,
                "document",
            ),
        ),
        (
            E("b-e1", "b-start", "b-in"),
            E("b-e2", "b-in", "b-candidate"),
            E("b-e3", "b-candidate", "b-width"),
            E("b-e4", "b-width", "b-check"),
            E("b-e5", "b-check", "b-purple", "có", "#82B366"),
            E("b-e6", "b-check", "b-fixed", "không", "#B85450", ((705, 675), (355, 675), (355, 800))),
            E("b-e7", "b-fixed", "b-out", "fallback", ((355, 885), (355, 1405), (620, 1405))),
            E("b-e8", "b-purple", "b-red"),
            E("b-e9", "b-red", "b-smooth"),
            E("b-e10", "b-smooth", "b-yellow"),
            E("b-e11", "b-yellow", "b-out"),
            E("b-e12", "b-out", "b-end"),
        ),
        page_h=1600,
    ),
    Page(
        "p05-tracking",
        "05 - Lane tracking theo thời gian",
        "5. Tracker lane: giữ biên làn ổn định theo thời gian",
        "05_lane_tracking_theo_thoi_gian.drawio",
        (
            N("t-start", "Bắt đầu cập nhật tracker", 680, 90, 250, 55, "terminator"),
            N("t-in", "Nhận biên làn hiện tại\nvà state frame trước", 620, 185, 370, 95, "data"),
            N("t-fit", "Fit ứng viên trái/phải", 620, 335, 370, 85, "predef"),
            N("t-score", "Đánh giá chất lượng ứng viên", 620, 475, 370, 85),
            N("t-ok", "Ứng viên đạt?", 705, 620, 200, 110, "decision"),
            N("t-accept", "Cập nhật theo phát hiện mới", 1060, 800, 330, 85),
            N("t-memory", "Cập nhật bộ nhớ bề rộng", 1060, 940, 330, 85),
            N("t-recover", "Dùng bộ nhớ hoặc fallback", 190, 800, 330, 85, "data"),
            N("t-combine", "Tạo biên làn ổn định", 620, 1080, 370, 90, "predef"),
            N("t-confidence", "Cập nhật độ tin cậy", 620, 1225, 370, 85),
            N("t-out", "Xuất biên làn đã track", 620, 1365, 370, 90, "data"),
            N("t-end", "Lane tracking sẵn sàng", 680, 1515, 250, 55, "terminator"),
            N(
                "t-note",
                "Luồng sau chỉ dùng snapshot kết quả. Không có truy cập chéo vào tracker state.",
                1050,
                1210,
                370,
                105,
                "document",
            ),
        ),
        (
            E("t-e1", "t-start", "t-in"),
            E("t-e2", "t-in", "t-fit"),
            E("t-e3", "t-fit", "t-score"),
            E("t-e4", "t-score", "t-ok"),
            E("t-e5", "t-ok", "t-accept", "có", "#82B366", ((905, 675), (1225, 675), (1225, 800))),
            E("t-e6", "t-accept", "t-memory"),
            E("t-e7", "t-memory", "t-combine", ((1225, 1025), (1225, 1125), (990, 1125))),
            E("t-e8", "t-ok", "t-recover", "không", "#B85450", ((705, 675), (355, 675), (355, 800))),
            E("t-e9", "t-recover", "t-combine", ((355, 885), (355, 1125), (620, 1125))),
            E("t-e10", "t-combine", "t-confidence"),
            E("t-e11", "t-confidence", "t-out"),
            E("t-e12", "t-out", "t-end"),
        ),
        page_h=1600,
    ),
    Page(
        "p06-control",
        "06 - Điều khiển",
        "6. Fusion control: ghép hướng lái và hành vi",
        "06_dieu_khien_lai.drawio",
        (
            N("c-start", "Bắt đầu fusion control", 675, 90, 260, 55, "terminator"),
            N("c-in", "Nhận kết quả lane/object", 620, 190, 370, 85, "data"),
            N("c-lane-ok", "Lane đủ tin cậy?", 250, 350, 210, 115, "decision"),
            N("c-steer", "Tính hướng lái", 210, 535, 290, 85),
            N("c-steer-safe", "Làm mượt và giới hạn lái", 210, 675, 290, 90, "predef"),
            N("c-steer-out", "Đề xuất hướng lái", 210, 825, 290, 80, "data"),
            N("c-obj-ok", "Có thông tin vật thể?", 1140, 350, 210, 115, "decision"),
            N("c-behavior", "Chọn hành vi\nstop, follow hoặc go", 1100, 535, 290, 95, "predef"),
            N("c-throttle", "Làm mượt ga/phanh", 1100, 685, 290, 90, "predef"),
            N("c-beh-out", "Đề xuất hành vi", 1100, 835, 290, 80, "data"),
            N("c-neutral", "Giữ lệnh an toàn mặc định", 655, 550, 300, 85, "data"),
            N("c-merge", "Ghép steering và throttle", 620, 1020, 370, 95, "predef"),
            N("c-out", "Xuất gói quyết định", 620, 1180, 370, 90, "data"),
            N("c-end", "Đưa vào hàng đợi output", 685, 1335, 240, 55, "terminator"),
            N(
                "c-note",
                "Safety supervisor vẫn kiểm tra lần cuối ở output trước khi gửi RC.",
                80,
                1110,
                420,
                100,
                "document",
            ),
        ),
        (
            E("c-e1", "c-start", "c-in"),
            E("c-e2", "c-in", "c-lane-ok", "lane", ((620, 232), (355, 232), (355, 350))),
            E("c-e3", "c-lane-ok", "c-steer", "có", "#82B366"),
            E("c-e4", "c-steer", "c-steer-safe"),
            E("c-e5", "c-steer-safe", "c-steer-out"),
            E("c-e6", "c-lane-ok", "c-neutral", "không", "#D6B656", ((460, 408), (655, 408), (655, 592))),
            E("c-e7", "c-in", "c-obj-ok", "object", ((990, 232), (1245, 232), (1245, 350))),
            E("c-e8", "c-obj-ok", "c-behavior", "có", "#82B366"),
            E("c-e9", "c-behavior", "c-throttle"),
            E("c-e10", "c-throttle", "c-beh-out"),
            E("c-e11", "c-obj-ok", "c-neutral", "không", "#D6B656", ((1140, 408), (955, 408), (955, 592))),
            E("c-e12", "c-steer-out", "c-merge", "steering", ((355, 905), (355, 1068), (620, 1068))),
            E("c-e13", "c-beh-out", "c-merge", "throttle", ((1245, 915), (1245, 1068), (990, 1068))),
            E("c-e14", "c-neutral", "c-merge", "default"),
            E("c-e15", "c-merge", "c-out"),
            E("c-e16", "c-out", "c-end"),
        ),
        page_h=1420,
    ),
    Page(
        "p07-object",
        "07 - Phát hiện và định vị vật thể",
        "7. Object worker: từ detection đến vị trí vật thể",
        "07_object_detection.drawio",
        (
            N("o-start", "Bắt đầu object worker", 675, 90, 260, 55, "terminator"),
            N("o-in", "Nhận frame từ hàng đợi object", 620, 185, 370, 85, "data"),
            N("o-pre", "Chuẩn bị ảnh cho mô hình object", 620, 325, 370, 80),
            N("o-model", "Chạy mô hình phát hiện vật thể", 620, 460, 370, 85, "predef"),
            N("o-decode", "Lọc và chuẩn hóa box", 620, 600, 370, 85),
            N("o-track", "Gán track cho vật thể", 620, 740, 370, 85, "predef"),
            N("o-corridor", "Nhận corridor từ lane", 190, 880, 330, 80, "data"),
            N("o-local", "Định vị từng detection", 620, 880, 370, 90, "predef"),
            N("o-type", "Loại vật thể?", 705, 1030, 200, 110, "decision"),
            N("o-road", "Nhánh vật thể trên đường", 190, 1210, 330, 85, "header"),
            N("o-road-depth", "Ước lượng khoảng cách", 190, 1345, 330, 85),
            N("o-road-path", "Kiểm tra nằm trong corridor", 190, 1485, 330, 85),
            N("o-front", "Chọn xe phía trước", 190, 1625, 330, 85, "data"),
            N("o-light", "Nhánh đèn giao thông", 1080, 1210, 330, 85, "header"),
            N("o-light-pos", "Ước lượng vị trí đèn", 1080, 1345, 330, 85),
            N("o-light-group", "Gom nhóm tín hiệu đèn", 1080, 1485, 330, 85, "predef"),
            N("o-light-out", "Đánh dấu để hiển thị", 1080, 1625, 330, 85, "data"),
            N("o-enrich", "Gộp pose vào detection", 620, 1785, 370, 90, "predef"),
            N("o-clean", "Dọn state cũ", 620, 1935, 370, 80),
            N("o-out", "Xuất gói kết quả object", 620, 2070, 370, 90, "data"),
            N("o-end", "Đưa vào hàng đợi fusion", 680, 2220, 250, 55, "terminator"),
            N(
                "o-note",
                "Road object có thể thành leading vehicle khi nằm trong corridor. Traffic light dùng cho hành vi dừng và hiển thị khoảng cách.",
                1080,
                1900,
                380,
                120,
                "document",
            ),
        ),
        (
            E("o-e1", "o-start", "o-in"),
            E("o-e2", "o-in", "o-pre"),
            E("o-e3", "o-pre", "o-model"),
            E("o-e4", "o-model", "o-decode"),
            E("o-e5", "o-decode", "o-track"),
            E("o-e6", "o-track", "o-local"),
            E("o-e7", "o-corridor", "o-local", "guidance"),
            E("o-e8", "o-local", "o-type"),
            E("o-e9", "o-type", "o-road", "vật thể đường", "#82B366", ((705, 1085), (355, 1085), (355, 1210))),
            E("o-e10", "o-road", "o-road-depth"),
            E("o-e11", "o-road-depth", "o-road-path"),
            E("o-e12", "o-road-path", "o-front"),
            E("o-e13", "o-front", "o-enrich", ((355, 1710), (355, 1830), (620, 1830))),
            E("o-e14", "o-type", "o-light", "đèn", "#D6B656", ((905, 1085), (1245, 1085), (1245, 1210))),
            E("o-e15", "o-light", "o-light-pos"),
            E("o-e16", "o-light-pos", "o-light-group"),
            E("o-e17", "o-light-group", "o-light-out"),
            E("o-e18", "o-light-out", "o-enrich", ((1245, 1710), (1245, 1830), (990, 1830))),
            E("o-e19", "o-enrich", "o-clean"),
            E("o-e20", "o-clean", "o-out"),
            E("o-e21", "o-out", "o-end"),
        ),
        page_h=2300,
    ),
    Page(
        "p08-behavior",
        "08 - Logic hành vi",
        "8. Behavior logic: ưu tiên stop, follow, speed cap, go",
        "08_behavior_logic.drawio",
        (
            N("g-start", "Bắt đầu behavior", 700, 90, 210, 55, "terminator"),
            N("g-in", "Nhận object state\nvà trạng thái lane", 620, 185, 370, 90, "data"),
            N("g-red", "Có đèn đỏ hiệu lực?", 705, 335, 200, 110, "decision"),
            N("g-stop", "Chọn STOP", 1080, 350, 300, 85, "data"),
            N("g-front", "Có xe phía trước?", 705, 515, 200, 110, "decision"),
            N("g-follow", "Chọn FOLLOW", 1080, 530, 300, 85, "data"),
            N("g-go", "Chọn GO", 260, 530, 300, 85, "data"),
            N("g-speed", "Có biển tốc độ?", 705, 700, 200, 110, "decision"),
            N("g-cap", "Giới hạn tốc độ", 1080, 720, 300, 85),
            N("g-raw", "Tạo trạng thái hành vi", 620, 885, 370, 90, "predef"),
            N("g-smooth", "Làm mượt theo thời gian", 620, 1035, 370, 90, "predef"),
            N("g-out", "Xuất behavior state", 620, 1185, 370, 85, "data"),
            N("g-end", "Behavior sẵn sàng", 695, 1330, 220, 55, "terminator"),
            N(
                "g-note",
                "Thứ tự ưu tiên: đèn đỏ trước, xe phía trước sau, cuối cùng mới áp biển tốc độ và chạy bình thường.",
                80,
                985,
                420,
                115,
                "document",
            ),
        ),
        (
            E("g-e1", "g-start", "g-in"),
            E("g-e2", "g-in", "g-red"),
            E("g-e3", "g-red", "g-stop", "có", "#B85450"),
            E("g-e4", "g-stop", "g-raw", ((1230, 435), (1230, 930), (990, 930))),
            E("g-e5", "g-red", "g-front", "không", "#82B366"),
            E("g-e6", "g-front", "g-follow", "có", "#D6B656"),
            E("g-e7", "g-follow", "g-speed", ((1230, 615), (1230, 755), (905, 755))),
            E("g-e8", "g-front", "g-go", "không", "#82B366"),
            E("g-e9", "g-go", "g-speed", ((410, 615), (410, 755), (705, 755))),
            E("g-e10", "g-speed", "g-cap", "có", "#D6B656"),
            E("g-e11", "g-cap", "g-raw", ((1230, 805), (1230, 930), (990, 930))),
            E("g-e12", "g-speed", "g-raw", "không", "#82B366"),
            E("g-e13", "g-raw", "g-smooth"),
            E("g-e14", "g-smooth", "g-out"),
            E("g-e15", "g-out", "g-end"),
        ),
        page_h=1420,
    ),
    Page(
        "p09-runtime",
        "09 - Chế độ chạy",
        "9. Runtime lifecycle: chuẩn bị, chọn chế độ, dừng sạch",
        "09_runtime_mode.drawio",
        (
            N("r-start", "Bắt đầu chương trình", 685, 90, 240, 55, "terminator"),
            N("r-config", "Đọc cấu hình và mode", 620, 190, 370, 85, "data"),
            N("r-feature", "Bật/tắt các nhánh xử lý", 620, 330, 370, 85),
            N("r-valid", "Cấu hình hợp lệ?", 705, 475, 200, 110, "decision"),
            N("r-error", "Dừng vì cấu hình lỗi", 180, 500, 320, 80, "data"),
            N("r-camera", "Chuẩn bị camera/video", 260, 690, 320, 85, "predef"),
            N("r-backend", "Chuẩn bị backend suy luận", 645, 690, 320, 85, "predef"),
            N("r-shape", "Chuẩn bị kích thước xử lý", 1030, 690, 320, 85),
            N("r-runtime", "Chọn kiểu runtime", 705, 885, 200, 110, "decision"),
            N("r-threaded", "Chạy multi-thread runtime", 1010, 1060, 330, 95, "predef"),
            N("r-single", "Chạy single-loop fallback", 260, 1060, 330, 95),
            N("r-monitor", "Theo dõi điều kiện dừng", 620, 1240, 370, 90, "decision"),
            N("r-clean", "Đóng tài nguyên và worker", 620, 1395, 370, 90),
            N("r-end", "Chương trình kết thúc", 685, 1545, 240, 55, "terminator"),
            N(
                "r-note",
                "Multi-thread là hướng chính cho runtime. Single-loop giữ lại để replay, debug và so sánh hành vi.",
                1050,
                1315,
                370,
                115,
                "document",
            ),
        ),
        (
            E("r-e1", "r-start", "r-config"),
            E("r-e2", "r-config", "r-feature"),
            E("r-e3", "r-feature", "r-valid"),
            E("r-e4", "r-valid", "r-error", "không", "#B85450"),
            E("r-e5", "r-error", "r-end", ((340, 580), (340, 1572), (685, 1572))),
            E("r-e6", "r-valid", "r-camera", "có", "#82B366", ((705, 530), (420, 530), (420, 690))),
            E("r-e7", "r-valid", "r-backend", "có", "#82B366"),
            E("r-e8", "r-valid", "r-shape", "có", "#82B366", ((905, 530), (1190, 530), (1190, 690))),
            E("r-e9", "r-camera", "r-runtime", ((420, 775), (420, 940), (705, 940))),
            E("r-e10", "r-backend", "r-runtime"),
            E("r-e11", "r-shape", "r-runtime", ((1190, 775), (1190, 940), (905, 940))),
            E("r-e12", "r-runtime", "r-threaded", "multi-thread", "#82B366"),
            E("r-e13", "r-runtime", "r-single", "fallback", "#D6B656"),
            E("r-e14", "r-threaded", "r-monitor", ((1175, 1155), (1175, 1285), (990, 1285))),
            E("r-e15", "r-single", "r-monitor", ((425, 1155), (425, 1285), (620, 1285))),
            E("r-e16", "r-monitor", "r-clean", "dừng", "#B85450"),
            E("r-e17", "r-clean", "r-end"),
        ),
        page_h=1640,
    ),
    Page(
        "p10-world",
        "10 - Fusion / world model",
        "10. Fusion: đồng bộ lane/object và tạo snapshot cho output",
        "10_world_model_fusion.drawio",
        (
            N("w-start", "Bắt đầu fusion", 700, 90, 210, 55, "terminator"),
            N("w-in", "Nhận kết quả lane/object", 620, 185, 370, 85, "data"),
            N("w-buffer", "Lưu kết quả theo frame", 620, 325, 370, 85),
            N("w-need", "Runtime cần nhánh nào?", 705, 470, 200, 110, "decision"),
            N("w-both", "Mode đầy đủ\nghép lane và object", 250, 650, 330, 95),
            N("w-single", "Mode một nhánh\nkhông chờ nhánh tắt", 1030, 650, 330, 95),
            N("w-stale", "Loại dữ liệu quá cũ", 250, 810, 330, 80),
            N("w-frame", "Tạo snapshot frame", 620, 810, 370, 85, "data"),
            N("w-control", "Tính quyết định tổng hợp", 620, 955, 370, 90, "predef"),
            N("w-out", "Xuất behavior packet", 620, 1105, 370, 90, "data"),
            N("w-q", "Đưa sang hàng đợi output", 620, 1255, 370, 80, "data"),
            N("w-end", "Fusion sẵn sàng", 700, 1395, 210, 55, "terminator"),
            N(
                "w-note",
                "Behavior packet hiện đóng vai trò snapshot của world/fusion cho output.",
                1050,
                1030,
                370,
                105,
                "document",
            ),
        ),
        (
            E("w-e1", "w-start", "w-in"),
            E("w-e2", "w-in", "w-buffer"),
            E("w-e3", "w-buffer", "w-need"),
            E("w-e4", "w-need", "w-both", "lane + object", "#82B366", ((705, 525), (415, 525), (415, 650))),
            E("w-e5", "w-both", "w-stale"),
            E("w-e6", "w-stale", "w-frame", ((415, 890), (415, 852), (620, 852))),
            E("w-e7", "w-need", "w-single", "một nhánh", "#D6B656", ((905, 525), (1195, 525), (1195, 650))),
            E("w-e8", "w-single", "w-frame", ((1195, 745), (1195, 852), (990, 852))),
            E("w-e9", "w-frame", "w-control"),
            E("w-e10", "w-control", "w-out"),
            E("w-e11", "w-out", "w-q"),
            E("w-e12", "w-q", "w-end"),
        ),
        page_h=1480,
    ),
    Page(
        "p11-safety",
        "11 - Safety supervisor / output",
        "11. Output thread: kiểm tra an toàn rồi phát hành kết quả",
        "11_safety_supervisor.drawio",
        (
            N("saf-start", "Bắt đầu output loop", 685, 90, 240, 55, "terminator"),
            N("saf-in", "Nhận behavior packet mới nhất", 620, 185, 370, 85, "data"),
            N("saf-check", "Kiểm tra an toàn cuối", 620, 325, 370, 90, "predef"),
            N("saf-command", "Tạo lệnh an toàn", 620, 480, 370, 85, "data"),
            N("saf-rc", "Nếu bật RC\ngửi steering và throttle", 190, 650, 330, 95),
            N("saf-render", "Nếu bật render\nvẽ lane, object, HUD", 640, 650, 330, 95, "predef"),
            N("saf-save", "Nếu bật output\nlưu video hoặc hiển thị", 1090, 650, 330, 95),
            N("saf-fps", "Cập nhật FPS và trạng thái", 620, 850, 370, 85),
            N("saf-stop", "Có yêu cầu dừng?", 705, 995, 200, 110, "decision"),
            N("saf-clean", "Đóng writer, cửa sổ, worker", 620, 1170, 370, 90),
            N("saf-end", "Output kết thúc", 695, 1320, 220, 55, "terminator"),
            N(
                "saf-note",
                "Trang này gom các nhánh optional thành ba khối song song để tránh dây chồng lên nhau.",
                80,
                1140,
                420,
                115,
                "document",
            ),
        ),
        (
            E("saf-e1", "saf-start", "saf-in"),
            E("saf-e2", "saf-in", "saf-check"),
            E("saf-e3", "saf-check", "saf-command"),
            E("saf-e4", "saf-command", "saf-rc", "RC", ((620, 522), (355, 522), (355, 650))),
            E("saf-e5", "saf-command", "saf-render", "render"),
            E("saf-e6", "saf-command", "saf-save", "video/window", ((990, 522), (1255, 522), (1255, 650))),
            E("saf-e7", "saf-rc", "saf-fps", ((355, 745), (355, 892), (620, 892))),
            E("saf-e8", "saf-render", "saf-fps"),
            E("saf-e9", "saf-save", "saf-fps", ((1255, 745), (1255, 892), (990, 892))),
            E("saf-e10", "saf-fps", "saf-stop"),
            E("saf-e11", "saf-stop", "saf-clean", "có", "#B85450"),
            E("saf-e12", "saf-clean", "saf-end"),
        ),
        page_h=1410,
    ),
    Page(
        "p12-threading",
        "12 - Kiến trúc multiple-threading",
        "12. Sơ đồ multi-thread: lane rõ ràng, hàng đợi rõ ràng",
        "12_threading_architecture.drawio",
        (
            N("th-start", "Runtime multi-thread", 685, 90, 250, 55, "terminator"),
            N("th-cap-head", "Luồng capture", 70, 200, 240, 55, "header"),
            N("th-lane-head", "Luồng lane", 380, 200, 240, 55, "header"),
            N("th-obj-head", "Luồng object", 690, 200, 240, 55, "header"),
            N("th-fusion-head", "Luồng fusion", 1000, 200, 240, 55, "header"),
            N("th-output-head", "Luồng output", 1310, 200, 240, 55, "header"),
            N("th-read", "Đọc frame mới", 85, 330, 210, 85, "predef"),
            N("th-lane-q", "Hàng đợi lane", 85, 500, 210, 75, "data"),
            N("th-object-q", "Hàng đợi object", 85, 670, 210, 75, "data"),
            N("th-lane", "Xử lý lane", 395, 500, 210, 85, "predef"),
            N("th-object", "Xử lý object", 705, 670, 210, 85, "predef"),
            N("th-lane-state", "Kết quả lane", 395, 1010, 210, 75, "data"),
            N("th-object-state", "Kết quả object", 705, 1010, 210, 75, "data"),
            N("th-sync", "Đồng bộ theo frame", 1015, 585, 210, 90, "predef"),
            N("th-fuse", "Tạo quyết định", 1015, 790, 210, 85, "predef"),
            N("th-behavior-q", "Hàng đợi output", 1015, 1010, 210, 75, "data"),
            N("th-safe", "Kiểm tra an toàn", 1325, 585, 210, 85, "predef"),
            N("th-render", "Hiển thị hoặc gửi RC", 1325, 790, 210, 90, "predef"),
            N("th-stop", "Điều phối dừng", 1325, 1010, 210, 85),
            N("th-roadmap1", "Hướng phát triển:\nsuy luận async", 380, 1210, 240, 90, "note"),
            N("th-roadmap2", "Hướng phát triển:\nưu tiên điều khiển", 690, 1210, 240, 90, "note"),
            N("th-roadmap3", "Hướng phát triển:\nđo sức khỏe hệ thống", 1000, 1210, 240, 90, "note"),
            N("th-end", "Các luồng dừng sạch", 685, 1390, 250, 55, "terminator"),
            N(
                "th-note",
                "Dây ngang là dữ liệu đi qua queue. Dây dọc trong cùng lane là nhiệm vụ nội bộ của thread đó.",
                70,
                1200,
                240,
                115,
                "document",
            ),
        ),
        (
            E("th-e1", "th-start", "th-read", points=((810, 165), (190, 165), (190, 330))),
            E("th-e2", "th-read", "th-lane-q", "frame"),
            E("th-e3", "th-read", "th-object-q", "frame"),
            E("th-e4", "th-lane-q", "th-lane"),
            E("th-e5", "th-object-q", "th-object"),
            E("th-e8", "th-lane", "th-lane-state"),
            E("th-e9", "th-object", "th-object-state"),
            E("th-e10", "th-lane-state", "th-sync", "kết quả lane", ((605, 1048), (955, 1048), (955, 630), (1015, 630))),
            E("th-e11", "th-object-state", "th-sync", "kết quả object", ((810, 1090), (990, 1090), (990, 655), (1015, 655))),
            E("th-e12", "th-sync", "th-fuse"),
            E("th-e13", "th-fuse", "th-behavior-q"),
            E("th-e14", "th-behavior-q", "th-safe", ((1225, 1048), (1285, 1048), (1285, 628), (1325, 628))),
            E("th-e15", "th-safe", "th-render"),
            E("th-e16", "th-render", "th-stop"),
            E("th-e17", "th-stop", "th-end", "dừng", "#B85450", ((1430, 1095), (1430, 1418), (935, 1418))),
        ),
        page_h=1480,
    ),
)


def main() -> None:
    FLOWCHART_DIR.mkdir(parents=True, exist_ok=True)
    modified = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    generated_diagrams = tuple(page_xml(page) for page in GENERATED_PAGES)
    combined_changed = write_mxfile_if_changed(OUT_PATH, generated_diagrams, modified)
    print(f"{'Updated' if combined_changed else 'Unchanged'}: {OUT_PATH.resolve()}")

    for page, diagram in zip(GENERATED_PAGES, generated_diagrams, strict=True):
        single_path = FLOWCHART_DIR / page.filename
        changed = write_mxfile_if_changed(single_path, (diagram,), modified)
        print(f"{'Updated' if changed else 'Unchanged'}: {single_path.resolve()}")


if __name__ == "__main__":
    main()
