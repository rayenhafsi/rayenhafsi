- 👋 Hi, I’m @rayenhafsi
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

    python-version: [3.9]
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
  4  
.github/workflows/release.yml
@@ -35,7 +35,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.9
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: 3.9
          cache: pip
@@ -61,7 +61,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
  2  
.github/workflows/test-linux.yml
@@ -69,7 +69,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.9
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: 3.9
      - name: Install Coverage
  2  
.github/workflows/test-mac.yml
@@ -30,7 +30,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
  2  
.github/workflows/test-win.yml
@@ -30,7 +30,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

