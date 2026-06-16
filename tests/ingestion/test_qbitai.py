from ingestion.qbitai import extract_qbitai_hot_news_items


def test_extract_qbitai_hot_news_items_from_sidebar():
    html = """
    <div class="content_right display_n">
      <div class="yaowen">
        <h3>热门文章</h3>
        <div class="picture_text">
          <a href="https://www.qbitai.com/2026/06/433731.html">
            <div class="text_box"><h4>GPT-5.6首批实测来了！精准狙击Mythos</h4></div>
            <div class="info">2026-06-10</div>
          </a>
        </div>
        <div class="picture_text">
          <a href="/2026/06/433798.html">
            <div class="text_box"><h4>英特尔锐炫 Pro B70 GPU亮相MPTS2026</h4></div>
            <div class="info">2026-06-10</div>
          </a>
        </div>
      </div>
    </div>
    """

    items = extract_qbitai_hot_news_items(html, "https://www.qbitai.com/category/news", limit=2)

    assert [item.rank for item in items] == [1, 2]
    assert items[0].title == "GPT-5.6首批实测来了！精准狙击Mythos"
    assert items[0].url == "https://www.qbitai.com/2026/06/433731.html"
    assert items[0].published_at == "2026-06-10"
    assert items[1].url == "https://www.qbitai.com/2026/06/433798.html"


def test_extract_qbitai_hot_news_items_falls_back_to_category_list():
    html = """
    <div class="article_list">
      <div class="picture_text">
        <div class="text_box">
          <h4><a href="/2026/06/435483.html">HuggingFace CEO力荐的HRM模型</a></h4>
        </div>
        <div class="info"><span class="time">前天 20:40</span></div>
      </div>
    </div>
    """

    items = extract_qbitai_hot_news_items(html, "https://www.qbitai.com/category/news", limit=3)

    assert len(items) == 1
    assert items[0].rank == 1
    assert items[0].title == "HuggingFace CEO力荐的HRM模型"
    assert items[0].published_at == "前天 20:40"
