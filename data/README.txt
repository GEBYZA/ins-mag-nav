示例 CSV 列: gx,gy,gz, ax,ay,az, mx,my,mz
单位: rad/s, m/s2, uT(任意一致单位)

生成完整数据(约数百行):
  python -m ins_mag_nav.generate_examples

本地试运行:
  python -m ins_mag_nav.run_from_csv ins_mag_nav/data/example_static.csv

在线运行:
  打开 https://colab.research.google.com/
  上传 INS_Mag_Nav_Colab.ipynb
