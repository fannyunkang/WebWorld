# initial_states.py — 预置的初始页面状态

GLOBAL_START_PAGE = """RootWebArea 'Global Start - Your Daily Portal', focused
\t[1] banner 'Top Header', visible
\t\t[2] link 'Set as Homepage', clickable, visible
\t\t[3] link 'Feedback', clickable, visible
\t\t[5] region 'Weather Widget', visible
\t\t\tStaticText 'New York, USA'
\t\t\t[6] image 'Sunny', visible
\t\t\tStaticText '24°C'
\t\t[8] link 'Sign In', clickable, visible
\t[10] region 'Search Area', visible
\t\t[11] image 'Global Start Logo', visible
\t\tStaticText 'Search the entire web'
\t\t[12] tablist 'Search Engine Selector', orientation='horizontal'
\t\t\t[13] tab 'Google', selected=True, clickable
\t\t\t[14] tab 'Bing', selected=False, clickable
\t\t\t[15] tab 'DuckDuckGo', selected=False, clickable
\t\t[18] combobox 'Web Search', clickable, visible, autocomplete='both', expanded=False
\t\t\t[19] textbox 'Type keywords or URL...', clickable, visible, editable, value=''
\t\t[20] button 'Search', clickable, visible
\t[30] navigation 'Category Bar', visible
\t\t[31] link 'Home', clickable, selected=True
\t\t[32] link 'News', clickable
\t\t[33] link 'Video', clickable
\t\t[34] link 'Shopping', clickable
\t\t[35] link 'Social', clickable
\t[50] main 'Site Directory', visible
\t\t[51] region 'Top Recommended', visible
\t\t\t[52] heading 'Most Popular', visible
\t\t\t[53] list 'Top Sites Grid', visible
\t\t\t\t[54] link 'Facebook', clickable
\t\t\t\t[56] link 'YouTube', clickable
\t\t\t\t[58] link 'Amazon', clickable
\t\t\t\t[60] link 'Twitter / X', clickable
\t\t\t\t[62] link 'Instagram', clickable
\t\t\t\t[64] link 'Wikipedia', clickable
\t\t\t\t[66] link 'Netflix', clickable
\t\t\t\t[68] link 'LinkedIn', clickable
\t\t[80] region 'News & Media', visible
\t\t\t[81] heading 'Latest News', visible
\t\t\t[82] link 'CNN', clickable
\t\t\t[83] link 'BBC', clickable
\t\t\t[84] link 'The Verge', clickable
\t\t[90] region 'Shopping', visible
\t\t\t[91] heading 'E-Commerce', visible
\t\t\t[92] link 'eBay', clickable
\t\t\t[93] link 'Walmart', clickable
\t\t\t[94] link 'Best Buy', clickable
\t[200] complementary 'Ads', visible
\t\t[201] image 'Ad: Travel to Japan'
\t\t[202] link 'Book Now', clickable
\t[300] contentinfo 'Footer', visible
\t\tStaticText '© 2026 Global Start Inc.'"""


HAO123_PAGE = """RootWebArea '123导航 - 中国首选上网首页', focused
\t[1] banner '顶部工具条', visible
\t\t[2] link '设为首页', clickable, visible
\t\t[3] link '保存到桌面', clickable, visible
\t\t[5] region '天气组件', visible
\t\t\tStaticText '北京'
\t\t\t[6] image '晴', visible
\t\t\tStaticText '5°C 优'
\t\t\tStaticText '农历腊月廿三'
\t\t[8] link '登录', clickable, visible
\t\t[9] link '注册', clickable, visible
\t[10] region '搜索专区', visible
\t\t[11] image '123导航 Logo', visible
\t\t[12] tablist '搜索引擎切换', orientation='horizontal'
\t\t\t[13] tab '百度', selected=True, clickable
\t\t\t[14] tab '搜狗', selected=False, clickable
\t\t\t[15] tab '必应', selected=False, clickable
\t\t\t[16] tab '谷歌', selected=False, clickable
\t\t[18] combobox '全网搜索', clickable, visible, autocomplete='both', expanded=False
\t\t\t[19] textbox '请输入搜索内容或网址...', clickable, visible, editable, value=''
\t\t[20] button '百度一下', clickable, visible
\t\t[21] link '热搜：春节放假安排', clickable, visible
\t[30] navigation '主要分类导航', visible
\t\t[31] link '首页', clickable, selected=True
\t\t[32] link '新闻', clickable
\t\t[33] link '视频', clickable
\t\t[34] link '电视剧', clickable
\t\t[35] link '购物', clickable
\t\t[36] link '游戏', clickable
\t\t[37] link '小说', clickable
\t[50] main '网站目录', visible
\t\t[51] region '名站推荐 (Top Sites)', visible
\t\t\t[52] list '名站网格', visible
\t\t\t\t[53] link '百度', clickable
\t\t\t\t\t[54] image '图标'
\t\t\t\t\tStaticText '百度'
\t\t\t\t[55] link '新浪', clickable
\t\t\t\t\t[56] image '图标'
\t\t\t\t\tStaticText '新浪'
\t\t\t\t[57] link '淘宝网', clickable
\t\t\t\t\t[58] image '图标'
\t\t\t\t\tStaticText '淘宝网'
\t\t\t\t[59] link '京东', clickable
\t\t\t\t\t[60] image '图标'
\t\t\t\t\tStaticText '京东'
\t\t\t\t[61] link '哔哩哔哩', clickable
\t\t\t\t\t[62] image '图标'
\t\t\t\t\tStaticText 'Bilibili'
\t\t\t\t[63] link '微博', clickable
\t\t\t\t\t[64] image '图标'
\t\t\t\t\tStaticText '微博'
\t\t\t\t[65] link '知乎', clickable
\t\t\t\t\t[66] image '图标'
\t\t\t\t\tStaticText '知乎'
\t\t\t\t[67] link '抖音', clickable
\t\t\t\t\t[68] image '图标'
\t\t\t\t\tStaticText '抖音'
\t\t\t\t[69] link '小红书', clickable
\t\t\t\t\t[70] image '图标'
\t\t\t\t\tStaticText '小红书'
\t\t\t\t[71] link '网易', clickable
\t\t\t\t\t[72] image '图标'
\t\t\t\t\tStaticText '网易'
\t\t[80] region '视频娱乐', visible
\t\t\t[81] heading '视频', visible
\t\t\t[82] link '爱奇艺', clickable
\t\t\t[83] link '腾讯视频', clickable
\t\t\t[84] link '优酷', clickable
\t\t\t[85] link '芒果TV', clickable
\t\t\t[86] link '西瓜视频', clickable
\t\t[100] region '购物导购', visible
\t\t\t[101] heading '购物', visible
\t\t\t[102] link '拼多多', clickable
\t\t\t[103] link '唯品会', clickable
\t\t\t[104] link '天猫', clickable
\t\t\t[105] link '苏宁易购', clickable
\t\t\t[106] link '当当网', clickable
\t\t[120] region '新闻资讯', visible
\t\t\t[121] heading '新闻', visible
\t\t\t[122] link '今日头条', clickable
\t\t\t[123] link '凤凰网', clickable
\t\t\t[124] link '澎湃新闻', clickable
\t\t\t[125] link '环球网', clickable
\t\t\t[126] link '人民网', clickable
\t\t[140] region '生活服务', visible
\t\t\t[141] heading '生活', visible
\t\t\t[142] link '美团', clickable
\t\t\t[143] link '58同城', clickable
\t\t\t[144] link '携程旅行', clickable
\t\t\t[145] link '去哪儿', clickable
\t\t\t[146] link '高德地图', clickable
\t\t\t[147] link '12306火车票', clickable
\t[200] complementary '侧边推荐', visible
\t\t[201] image '热门游戏广告: 贪玩蓝月'
\t\t[202] link '点击开始游戏', clickable
\t\t[203] heading '猜你喜欢'
\t\t[204] link '9.9元包邮', clickable
\t[300] contentinfo '页脚', visible
\t\tStaticText '© 2026 123导航'
\t\t[301] link '关于我们', clickable
\t\t[302] link '意见反馈', clickable
\t\t[303] link '京ICP证000000号', clickable"""


GOOGLE_PAGE = """RootWebArea 'Google', focused
\t[1] navigation 'header', visible
\t\t[2] link 'Gmail', clickable, visible
\t\t[3] link 'Images', clickable, visible
\t\t[4] button 'Google apps', clickable, visible, hasPopup='menu', expanded=False
\t\t\t[5] image ''
\t\t[6] link 'Sign in', clickable, visible
\t\t\t[7] button 'Sign in'
\t[10] main '', visible
\t\t[11] image 'Google', visible
\t\t[15] combobox 'Search', clickable, visible, autocomplete='both', expanded=False, hasPopup='listbox'
\t\t\t[16] textbox 'Search', clickable, visible, editable, multiline=False
\t\t\t[18] button 'Clear', clickable, visible, hidden=True
\t\t\t[20] button 'Search by voice', clickable, visible
\t\t\t\t[21] image ''
\t\t\t[22] button 'Search by image', clickable, visible
\t\t\t\t[23] image ''
\t\t[25] button 'Google Search', clickable, visible
\t\t[26] button "I'm Feeling Lucky", clickable, visible
\t\t[30] region 'Google offered in:', visible
\t\t\tStaticText 'Google offered in: '
\t\t\t[31] link 'Français', clickable, visible
\t\t\t[32] link '中文(简体)', clickable, visible
\t[40] contentinfo 'footer', visible
\t\tStaticText 'United Kingdom'
\t\t[42] link 'About', clickable, visible
\t\t[43] link 'Advertising', clickable, visible
\t\t[44] link 'Business', clickable, visible
\t\t[45] link 'How Search works', clickable, visible
\t\t[50] link 'Privacy', clickable, visible
\t\t[51] link 'Terms', clickable, visible
\t\t[52] button 'Settings', clickable, visible, hasPopup='menu', expanded=False"""


# ── 页面名称 → 内容的映射，命令行通过 --page 选择 ──

PAGES = {
    "global-start": GLOBAL_START_PAGE,
    "hao123": HAO123_PAGE,
    "google": GOOGLE_PAGE,
}

# 默认页面
DEFAULT_PAGE = "global-start"
