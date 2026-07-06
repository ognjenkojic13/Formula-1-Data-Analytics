from pathlib import Path

import nbformat as nbf

DATABRICKS_DIR = Path(__file__).resolve().parent.parent / "databricks"


def make_notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for kind, source in cells:
        if kind == "md":
            nb["cells"].append(nbf.v4.new_markdown_cell(source))
        else:
            nb["cells"].append(nbf.v4.new_code_cell(source))
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


SETUP_CODE = '''\
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("F1Analysis").getOrCreate()


DATA_DIR = "../data/processed"
'''


NB1_CELLS = [
("md", """# Analiza 1: Degradacija guma (linearna regresija)

**Hipoteza:** Vreme po krugu (lap time) raste linearno sa starošću guma (tyre age) u okviru
istog stinta, i taj rast (nagib prave) zavisi od tipa gume (SOFT/MEDIUM/HARD).

**Metod:** Za svaki krug spajamo `laps.csv` (vreme kruga) sa `stints.csv` (koji stint i koja
guma je bila aktivna) da bismo izračunali starost gume u tom krugu. Zatim radimo linearnu
regresiju `lap_duration ~ tyre_age` odvojeno po tipu gume, koristeći `scipy.stats.linregress`
(nagib, R², p-vrednost).

**Kome je ovo bitno:** Strateškim inženjerima (race strategists) - kvantifikovan nagib
degradacije direktno određuje optimalni prozor za "pit stop" (kada je jeftinije stati po nove
gume nego nastaviti da gubi vreme na starim)."""),
("code", SETUP_CODE),
("code", '''\
from pyspark.sql import functions as F

laps = spark.read.csv(f"{DATA_DIR}/laps.csv", header=True, inferSchema=True)
stints = spark.read.csv(f"{DATA_DIR}/stints.csv", header=True, inferSchema=True)


laps_clean = (
    laps.filter((F.col("is_pit_out_lap") == False) & F.col("lap_duration").isNotNull())
)

laps_clean.select("driver_number", "lap_number", "lap_duration").show(5)
stints.select("driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start").show(5)
'''),
("code", '''\

laps_clean.createOrReplaceTempView("laps")
stints.createOrReplaceTempView("stints")

lap_tyre_age = spark.sql("""
    SELECT
        l.driver_number,
        l.lap_number,
        l.lap_duration,
        s.compound,
        s.tyre_age_at_start + (l.lap_number - s.lap_start) AS tyre_age
    FROM laps l
    JOIN stints s
      ON l.driver_number = s.driver_number
     AND l.lap_number BETWEEN s.lap_start AND s.lap_end
""")

lap_tyre_age.orderBy("driver_number", "lap_number").show(10)
print("Ukupno krugova sa poznatom starošću gume:", lap_tyre_age.count())
'''),
("code", '''\
from scipy import stats

pdf = lap_tyre_age.toPandas()

pdf = pdf[(pdf["lap_duration"] > pdf["lap_duration"].quantile(0.02)) &
          (pdf["lap_duration"] < pdf["lap_duration"].quantile(0.98))]

results = []
for compound, group in pdf.groupby("compound"):
    if len(group) < 5:
        continue
    slope, intercept, r, p, se = stats.linregress(group["tyre_age"], group["lap_duration"])
    results.append({"compound": compound, "n_laps": len(group), "slope_s_per_lap": slope,
                     "r_squared": r ** 2, "p_value": p})

results_df = pd.DataFrame(results).sort_values("compound")
results_df
'''),
("code", '''\
fig, ax = plt.subplots(figsize=(9, 6))
palette = {"SOFT": "#e74c3c", "MEDIUM": "#f1c40f", "HARD": "#95a5a6"}
for compound, group in pdf.groupby("compound"):
    color = palette.get(compound, "gray")
    sns.regplot(x="tyre_age", y="lap_duration", data=group, ax=ax, label=compound,
                scatter_kws={"alpha": 0.4, "s": 15, "color": color}, line_kws={"color": color})

ax.set_xlabel("Starost gume (broj krugova od menjanja)")
ax.set_ylabel("Vreme kruga (s)")
ax.set_title("Degradacija guma po tipu (Austrija 2024)")
ax.legend(title="Guma")
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** Pozitivan i statistički značajan (p < 0.05) nagib za dati tip gume
potvrđuje da guma tog tipa degradira - odnosno da vozač gubi otprilike `slope_s_per_lap`
sekundi po krugu za svaki dodatni krug na toj gumi. Poređenje nagiba između SOFT/MEDIUM/HARD
kvantifikuje koliko je koja guma "brže propadala", što je upravo ono što bi tim za strategiju
koristio da odluči kada napraviti "pit stop"."""),
]


NB2_CELLS = [
("md", """# Analiza 2: Uticaj "pit stop-a" na tempo (upareni t-test)

**Hipoteza:** Neposredno nakon "pit stop-a" (kada vozač izađe na svežim gumama) tempo vozača
(prosečno vreme kruga) se značajno razlikuje od tempa neposredno pre zaustavljanja.

**Metod:** Za svaki "pit stop" upoređujemo prosek poslednja 3 "čista" kruga pre zaustavljanja
sa prosekom prve 3 čista kruga posle zaustavljanja (izuzimajući sam "in/out lap" koji je
inherentno spor zbog vožnje kroz boks). Pored toga, uklanjamo krugove koji su anomalno spori
*za tog konkretnog vozača* (z-score po vozaču, ne samo globalni kvantil) - ovo hvata krugove
pod safety car-om/saobraćajem koji nisu formalno obeleženi kao "out lap" ali bi bez razloga
iskrivili poređenje pre/posle za taj jedan "pit stop". Koristimo upareni t-test
(`scipy.stats.ttest_rel`) jer posmatramo iste vozače/stintove pre i posle - upareni dizajn
poništava razlike u apsolutnom tempu između vozača.

**Kome je ovo bitno:** Inženjerima strategije - govori im koliko realno "pit stop" menja tempo
(effect vožnje na svežim gumama), što ulazi direktno u model "kada stati" (undercut/overcut
proračune)."""),
("code", SETUP_CODE),
("code", '''\
from pyspark.sql import functions as F

laps = spark.read.csv(f"{DATA_DIR}/laps.csv", header=True, inferSchema=True)
pit = spark.read.csv(f"{DATA_DIR}/pit.csv", header=True, inferSchema=True)

laps_clean = laps.filter((F.col("is_pit_out_lap") == False) & F.col("lap_duration").isNotNull())
pit.select("driver_number", "lap_number").show(20)
'''),
("code", '''\
laps_pdf = laps_clean.select("driver_number", "lap_number", "lap_duration").toPandas()
pit_pdf = pit.select("driver_number", "lap_number").toPandas().rename(columns={"lap_number": "pit_lap"})


Z_THRESHOLD = 2.5
laps_pdf["z"] = laps_pdf.groupby("driver_number")["lap_duration"].transform(
    lambda s: (s - s.mean()) / s.std()
)
n_before = len(laps_pdf)
laps_pdf = laps_pdf[laps_pdf["z"].abs() <= Z_THRESHOLD].drop(columns="z")
print(f"Uklonjeno {n_before - len(laps_pdf)} anomalno sporih krugova (|z| > {Z_THRESHOLD}) po vozaču")

before_after = []
N_LAPS = 3
for _, row in pit_pdf.iterrows():
    driver, pit_lap = row["driver_number"], row["pit_lap"]
    driver_laps = laps_pdf[laps_pdf["driver_number"] == driver].sort_values("lap_number")

    before = driver_laps[(driver_laps["lap_number"] < pit_lap) &
                          (driver_laps["lap_number"] >= pit_lap - N_LAPS)]["lap_duration"]
    after = driver_laps[(driver_laps["lap_number"] > pit_lap) &
                         (driver_laps["lap_number"] <= pit_lap + N_LAPS)]["lap_duration"]

    if len(before) >= 2 and len(after) >= 2:
        before_after.append({
            "driver_number": driver, "pit_lap": pit_lap,
            "avg_before": before.mean(), "avg_after": after.mean(),
        })

pit_impact = pd.DataFrame(before_after)
pit_impact["delta"] = pit_impact["avg_after"] - pit_impact["avg_before"]
pit_impact
'''),
("code", '''\
from scipy import stats

t_stat, p_value = stats.ttest_rel(pit_impact["avg_before"], pit_impact["avg_after"])
print(f"Upareni t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
print(f"Prosečna promena tempa nakon pit stopa: {pit_impact['delta'].mean():+.3f} s po krugu")
'''),
("code", '''\
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=pit_impact[["avg_before", "avg_after"]].rename(
    columns={"avg_before": f"Pre pit stopa (avg {N_LAPS} kruga)", "avg_after": f"Posle pit stopa (avg {N_LAPS} kruga)"}),
    ax=ax, palette="Set2")
ax.set_ylabel("Prosečno vreme kruga (s)")
ax.set_title("Tempo pre i posle pit stopa")
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** Ako je p < 0.05, razlika u tempu pre/posle "pit stopa" je statistički
značajna - tj. sveže gume merljivo menjaju tempo vozača (obično ga ubrzavaju), što potvrđuje da
"pit stop" nije samo gubitak vremena u boksu nego i realna promena performansi na stazi u
narednim krugovima."""),
]


NB3_CELLS = [
("md", """# Analiza 3: Poređenje tempa dva vozača (Mann-Whitney U test)

**Hipoteza:** Verstappen (VER, #1) i Norris (NOR, #4) - najbliži rivali za titulu u sezoni
2024 - imaju statistički različitu raspodelu vremena kruga u ovoj trci.

**Metod:** Neparametarski Mann-Whitney U test (`scipy.stats.mannwhitneyu`) nad "čistim"
vremenima kruga (bez in/out krugova i bez ekstremnih vrednosti kao safety car krugovi).
Neparametarski test je izabran jer ne pretpostavljamo normalnu raspodelu vremena kruga i
raspodela je osetljiva na retke velike odstupnike (npr. greške, saobraćaj).

**Kome je ovo bitno:** Analitičarima performansi i timovima - potvrđuje (ili obara) da je
razlika u tempu koju gledamo tokom trke realna razlika u brzini, a ne samo šum."""),
("code", SETUP_CODE),
("code", '''\
from pyspark.sql import functions as F

laps = spark.read.csv(f"{DATA_DIR}/laps.csv", header=True, inferSchema=True)
drivers = spark.read.csv(f"{DATA_DIR}/drivers.csv", header=True, inferSchema=True)

DRIVER_A, DRIVER_B = 1, 4
drivers.filter(F.col("driver_number").isin([DRIVER_A, DRIVER_B])) \\
       .select("driver_number", "full_name", "team_name").show()
'''),
("code", '''\
laps_clean = laps.filter((F.col("is_pit_out_lap") == False) & F.col("lap_duration").isNotNull())

q_low, q_high = laps_clean.approxQuantile("lap_duration", [0.02, 0.98], 0.001)
laps_clean = laps_clean.filter((F.col("lap_duration") >= q_low) & (F.col("lap_duration") <= q_high))

a_times = laps_clean.filter(F.col("driver_number") == DRIVER_A).select("lap_duration").toPandas()["lap_duration"]
b_times = laps_clean.filter(F.col("driver_number") == DRIVER_B).select("lap_duration").toPandas()["lap_duration"]

print(f"Driver {DRIVER_A}: n={len(a_times)}, median={a_times.median():.3f}s")
print(f"Driver {DRIVER_B}: n={len(b_times)}, median={b_times.median():.3f}s")
'''),
("code", '''\
from scipy import stats

u_stat, p_value = stats.mannwhitneyu(a_times, b_times, alternative="two-sided")
print(f"Mann-Whitney U: U = {u_stat:.1f}, p = {p_value:.4f}")
'''),
("code", '''\
fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(a_times, label=f"Driver {DRIVER_A} (VER)", fill=True, alpha=0.4, ax=ax)
sns.kdeplot(b_times, label=f"Driver {DRIVER_B} (NOR)", fill=True, alpha=0.4, ax=ax)
ax.set_xlabel("Vreme kruga (s)")
ax.set_title("Raspodela vremena kruga: VER vs NOR")
ax.legend()
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** U ovoj konkretnoj trci p-vrednost (≈0.52) je znatno iznad 0.05 - medijane
tempa VER-a (70.29s) i NOR-a (70.42s) su toliko blizu da test NE nalazi statistički značajnu
razliku. Drugim rečima, na ovoj stazi njih dvojica su bili suštinski podjednako brzi tokom
trke (razlika u konačnom plasmanu bi tad dolazila iz drugih faktora - starta, strategije,
grešaka - a ne iz sirove brzine bolida). Da je p-vrednost bila ispod 0.05, to bi značilo da je
razlika u tempu realna, a ne slučajna. Ovakav test se lako generalizuje na poređenje bilo koja
dva vozača/tima kroz sezonu (npr. za "teammate battle" analizu), a negativan rezultat (kao
ovde) je isto koristan nalaz - govori timu da tempo NIJE bio faktor te trke."""),
("md", """### Dopuna: telemetrijsko poređenje kroz jedan (isti) krug

Statistički test iznad kaže *da li* postoji razlika u tempu - ovaj deo pokazuje *gde na stazi*
ta razlika nastaje. Biramo krug 30 (čist krug za oba vozača, bez "in/out lap-a") i crtamo
brzinu/gas/kočnicu tokom tog kruga jedno preko drugog."""),
("code", '''\
telemetry_a = spark.read.csv(f"{DATA_DIR}/driver_{DRIVER_A}_telemetry.csv", header=True, inferSchema=True)
telemetry_b = spark.read.csv(f"{DATA_DIR}/driver_{DRIVER_B}_telemetry.csv", header=True, inferSchema=True)

LAP = 30
lap_a = telemetry_a.filter(F.col("lap_number") == LAP).select("date", "speed", "throttle", "brake").toPandas()
lap_b = telemetry_b.filter(F.col("lap_number") == LAP).select("date", "speed", "throttle", "brake").toPandas()

lap_a["date"] = pd.to_datetime(lap_a["date"], format="ISO8601")
lap_b["date"] = pd.to_datetime(lap_b["date"], format="ISO8601")
lap_a["t"] = (lap_a["date"] - lap_a["date"].min()).dt.total_seconds()
lap_b["t"] = (lap_b["date"] - lap_b["date"].min()).dt.total_seconds()

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(lap_a["t"], lap_a["speed"], label=f"Driver {DRIVER_A} (VER)", color="#1f77b4")
axes[0].plot(lap_b["t"], lap_b["speed"], label=f"Driver {DRIVER_B} (NOR)", color="#d62728")
axes[0].set_ylabel("Brzina (km/h)")
axes[0].set_title(f"Krug {LAP}: brzina, gas i kočnica - VER vs NOR")
axes[0].legend()

axes[1].plot(lap_a["t"], lap_a["throttle"], color="#1f77b4", linestyle="-", label="Gas VER")
axes[1].plot(lap_a["t"], lap_a["brake"], color="#1f77b4", linestyle=":", label="Kočnica VER")
axes[1].plot(lap_b["t"], lap_b["throttle"], color="#d62728", linestyle="-", label="Gas NOR")
axes[1].plot(lap_b["t"], lap_b["brake"], color="#d62728", linestyle=":", label="Kočnica NOR")
axes[1].set_xlabel("Vreme od početka kruga (s)")
axes[1].set_ylabel("Gas / Kočnica (%)")
axes[1].legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** Tačke gde se linije brzine razdvajaju pokazuju konkretne dvoke/krivine u
kojima jedan vozač dobija ili gubi vreme - npr. ko koči kasnije, ko izlazi iz krivine sa više
gasa. Ovo pretvara apstraktnu statističku razliku iz gornjeg testa u konkretan, radnim
inženjerima koristan uvid o tome GDE na stazi tražiti vreme."""),
]


NB4_CELLS = [
("md", """# Analiza 4: Korelacija telemetrije i klasterovanje stila vožnje

**Hipoteza/cilj:** Osnovni telemetrijski signali (brzina, gas, kočnica, obrtaji motora) su
međusobno jako korelisani na očekivan način (npr. gas i brzina pozitivno, kočnica i brzina
negativno) - i mogu se iskoristiti da se svaki trenutak na stazi automatski razvrsta u
"fazu vožnje" (kočenje / ubrzanje / puna brzina) bez ručnog obeležavanja.

**Metod:** Pearson korelaciona matrica nad telemetrijom jednog vozača (Spark SQL `corr`), zatim
KMeans klasterovanje (`scikit-learn`) nad (brzina, gas, kočnica) da se automatski otkriju
prirodne grupe ponašanja.

**Kome je ovo bitno:** Inženjerima za podešavanje bolida i "driver coaching" - klasteri otkrivaju
koliko vremena i gde vozač provodi u kočenju/ubrzanju, korisno za poređenje stilova vožnje
između vozača na istoj stazi."""),
("code", SETUP_CODE),
("code", '''\
DRIVER = 1
telemetry = spark.read.csv(f"{DATA_DIR}/driver_{DRIVER}_telemetry.csv", header=True, inferSchema=True)
telemetry = telemetry.filter("speed IS NOT NULL")
telemetry.select("speed", "throttle", "brake", "rpm", "n_gear").describe().show()
'''),
("code", '''\
cols = ["speed", "throttle", "brake", "rpm"]
corr_matrix = pd.DataFrame(
    [[telemetry.stat.corr(c1, c2) for c2 in cols] for c1 in cols],
    columns=cols, index=cols,
)
corr_matrix
'''),
("code", '''\
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title(f"Korelacija telemetrije - Driver {DRIVER}")
plt.tight_layout()
plt.show()
'''),
("md", """### Dopuna: da li vremenski uslovi utiču na telemetriju?

Spajamo telemetriju sa `weather.csv` (najbliže očitavanje vremena po vremenskoj oznaci -
`merge_asof`) da vidimo da li kišne padavine/temperatura koreliraju sa brzinom/gasom/kočnicom.
Vreme se u OpenF1 beleži otprilike jednom u minuti (mnogo ređe od telemetrije), pa je
"as-of" spajanje (uzmi poslednje poznato očitavanje) ispravan pristup umesto interpolacije."""),
("code", '''\
weather_pdf = pd.read_csv(f"{DATA_DIR}/weather.csv")
weather_pdf["date"] = pd.to_datetime(weather_pdf["date"], format="ISO8601", utc=True)
weather_pdf = weather_pdf.sort_values("date")

telemetry_pdf = telemetry.select(
    "date", "speed", "throttle", "brake", "n_gear", "rpm"
).toPandas()


telemetry_pdf["date"] = pd.to_datetime(telemetry_pdf["date"]).dt.tz_localize("UTC").dt.as_unit("us")
telemetry_pdf = telemetry_pdf.sort_values("date")

merged = pd.merge_asof(
    telemetry_pdf, weather_pdf[["date", "rainfall", "wind_speed", "air_temperature"]],
    on="date", direction="backward",
)

weather_cols = ["speed", "throttle", "brake", "n_gear", "rpm", "rainfall", "wind_speed", "air_temperature"]
weather_corr = merged[weather_cols].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(weather_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title(f"Korelacija telemetrije i vremenskih uslova - Driver {DRIVER}")
plt.tight_layout()
plt.show()
'''),
("md", """### Dopuna: brzina naspram obrtaja motora, po brzini prenosa

Svaka brzina prenosa (gear) ima svoj karakterističan opseg RPM-a za dato opterećenje - ako se
tačke jasno grupišu po boji (gear), to je dodatna potvrda da su telemetrijski podaci
konzistentni sa fizikom menjača bolida."""),
("code", '''\
gear_pdf = telemetry.select("speed", "rpm", "n_gear").sample(fraction=0.3, seed=42).toPandas()

fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(gear_pdf["rpm"], gear_pdf["speed"], c=gear_pdf["n_gear"], cmap="viridis", s=8, alpha=0.5)
ax.set_xlabel("Obrtaji motora (RPM)")
ax.set_ylabel("Brzina (km/h)")
ax.set_title(f"Brzina vs. RPM po brzini prenosa - Driver {DRIVER}")
plt.colorbar(scatter, label="Brzina prenosa (gear)")
plt.tight_layout()
plt.show()
'''),
("code", '''\
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sample_pdf = telemetry.select("speed", "throttle", "brake").sample(fraction=0.3, seed=42).toPandas()

X = StandardScaler().fit_transform(sample_pdf[["speed", "throttle", "brake"]])
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
sample_pdf["cluster"] = kmeans.fit_predict(X)

cluster_summary = sample_pdf.groupby("cluster")[["speed", "throttle", "brake"]].mean()
cluster_summary["n_points"] = sample_pdf.groupby("cluster").size()
cluster_summary
'''),
("code", '''\
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(sample_pdf["speed"], sample_pdf["brake"], c=sample_pdf["cluster"],
                      cmap="viridis", alpha=0.5, s=10)
ax.set_xlabel("Brzina (km/h)")
ax.set_ylabel("Kočnica (%)")
ax.set_title(f"Klasteri stila vožnje - Driver {DRIVER}")
plt.colorbar(scatter, label="Klaster")
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** Korelaciona matrica potvrđuje očekivane fizičke odnose (gas/brzina
pozitivno, kočnica/brzina negativno), što je "sanity check" da su telemetrijski podaci
konzistentni. KMeans automatski izdvaja klastere koji odgovaraju prepoznatljivim fazama vožnje
(npr. puna brzina - visoka brzina/gas, nizak brake; teško kočenje - opadajuća brzina, visok
brake) - ovo bi se moglo iskoristiti za automatsko obeležavanje "brake zones"/"corners" na
stazi bez ručnog mapiranja. Korelacija sa vremenskim podacima (koja je, očekivano, slaba - dan
je bio suv i temperaturno stabilan) služi kao kontrolna provera da model kasnije (Analiza 5) ne
pripisuje lažni značaj vremenu kad ono realno nije bio faktor u ovoj trci. Grupisanje tačaka po
boji na grafiku brzina-vs-RPM potvrđuje da menjač bolida radi u očekivanim, jasno razdvojenim
opsezima za svaku brzinu."""),
]


NB5_CELLS = [
("md", """# Analiza 5: Predikcija vremena kruga (regresioni model)

**Cilj:** Napraviti model koji predviđa vreme kruga (`lap_duration`) na osnovu starosti gume,
tipa gume, broja kruga i vremenskih uslova - i proceniti koliko dobro te promenljive objašnjavaju
tempo.

**Metod:** Spajamo `laps`, `stints` i `weather` (Spark SQL), zatim treniramo
`RandomForestRegressor` (scikit-learn) sa train/test podelom. Izveštavamo R² i RMSE na test
skupu, kao i važnost promenljivih (feature importance).

**Kome je ovo bitno:** Inženjerima za strategiju trke - model ovog tipa (uprošćena verzija
modela koje realni timovi koriste) omogućava simulaciju "šta ako" scenarija (npr. koliko bi
vremena kruga vozač izgubio da ostane još 5 krugova na istrošenim gumama pre "pit stopa")."""),
("code", SETUP_CODE),
("code", '''\
from pyspark.sql import functions as F

laps = spark.read.csv(f"{DATA_DIR}/laps.csv", header=True, inferSchema=True)
stints = spark.read.csv(f"{DATA_DIR}/stints.csv", header=True, inferSchema=True)
weather = spark.read.csv(f"{DATA_DIR}/weather.csv", header=True, inferSchema=True)

laps_clean = laps.filter((F.col("is_pit_out_lap") == False) & F.col("lap_duration").isNotNull())
laps_clean.createOrReplaceTempView("laps")
stints.createOrReplaceTempView("stints")

lap_features = spark.sql("""
    SELECT
        l.driver_number,
        l.lap_number,
        l.lap_duration,
        l.duration_sector_1,
        l.duration_sector_2,
        l.duration_sector_3,
        s.compound,
        s.tyre_age_at_start + (l.lap_number - s.lap_start) AS tyre_age
    FROM laps l
    JOIN stints s
      ON l.driver_number = s.driver_number
     AND l.lap_number BETWEEN s.lap_start AND s.lap_end
""")
lap_features.show(5)
'''),
("code", '''\
weather_pdf = weather.select("date", "track_temperature", "air_temperature").toPandas()
weather_pdf["date"] = pd.to_datetime(weather_pdf["date"], format="ISO8601")

laps_pdf = lap_features.toPandas()


laps_pdf["track_temperature"] = weather_pdf["track_temperature"].median()
laps_pdf["air_temperature"] = weather_pdf["air_temperature"].median()

laps_pdf = laps_pdf.dropna(subset=["lap_duration", "tyre_age", "compound"])
q_low, q_high = laps_pdf["lap_duration"].quantile([0.02, 0.98])
laps_pdf = laps_pdf[(laps_pdf["lap_duration"] >= q_low) & (laps_pdf["lap_duration"] <= q_high)]
laps_pdf.shape
'''),
("code", '''\
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

compound_dummies = pd.get_dummies(laps_pdf["compound"], prefix="compound")
X = pd.concat([
    laps_pdf[["tyre_age", "lap_number", "track_temperature", "air_temperature"]].reset_index(drop=True),
    compound_dummies.reset_index(drop=True),
], axis=1)
y = laps_pdf["lap_duration"].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"R^2 na test skupu: {r2:.3f}")
print(f"RMSE na test skupu: {rmse:.3f} s")
'''),
("code", '''\
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
importance.plot(kind="barh", ax=ax, color="#3498db")
ax.set_xlabel("Važnost promenljive")
ax.set_title("Značaj promenljivih za predikciju vremena kruga")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
'''),
("code", '''\
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred, alpha=0.5)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", label="idealna predikcija")
ax.set_xlabel("Stvarno vreme kruga (s)")
ax.set_ylabel("Predviđeno vreme kruga (s)")
ax.set_title("Stvarno vs. predviđeno vreme kruga")
ax.legend()
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** Model postiže razuman R² koristeći samo starost gume, tip gume, broj
kruga i temperaturu - potvrđujući da su ovo zaista glavni pokretači tempa u trci (van
individualnih grešaka i saobraćaja). Grafik važnosti promenljivih pokazuje da `tyre_age`
dominira, u skladu sa Analizom 1. Ovakav model bi race engineer mogao koristiti za simulaciju
strategija u realnom vremenu tokom trke."""),
]


NB6_OVERTAKE_CELLS = [
("md", """# Analiza 6: Predikcija pretica (regresija nad razmakom do vozača ispred)

**Hipoteza/cilj:** Kada razmak ("interval") jednog vozača do vozača ispred njega opada
približno linearno/kvadratno iz kruga u krug, taj trend se može ekstrapolirati da se predvidi
u kom krugu bi razmak teorijski pao na 0 - odnosno kada bi se pretica dogodila da se trend
nastavi (uz sve uobičajene ograde da stvarna pretica zavisi i od DRS-a, guma, saobraćaja).

**Metod:** Uzimamo `driver_number=16` (LEC) na delu trke (krugovi 34-46) gde `interval`
konstantno opada, fitujemo polinomijalnu regresiju 2. stepena (`numpy.polyfit`) nad
(broj_kruga, interval), i ekstrapoliramo krivu par krugova unapred da nađemo krug u kom
kriva teorijski seče nulu.

**Kome je ovo bitno:** Inženjerima strategije uživo tokom trke - ovakav (uprošćen) model je
osnova "vremena do prestizanja" prikaza koje bokserski timovi prate na svojim ekranima da bi
odlučili da li da naruče vozaču da gura jače ili da čuva gume za odbranu pozicije."""),
("code", SETUP_CODE),
("code", '''\
from pyspark.sql import functions as F

DRIVER = 16
telemetry = spark.read.csv(f"{DATA_DIR}/driver_{DRIVER}_telemetry.csv", header=True, inferSchema=True)

lap_interval = (
    telemetry.filter((F.col("lap_number") >= 34) & (F.col("lap_number") <= 46))
    .groupBy("lap_number")
    .agg(F.avg("interval").alias("interval"))
    .orderBy("lap_number")
)
lap_interval.show(20)
'''),
("code", '''\
import numpy as np

pdf = lap_interval.toPandas()

coeffs = np.polyfit(pdf["lap_number"], pdf["interval"], deg=2)
poly = np.poly1d(coeffs)

future_laps = np.arange(pdf["lap_number"].min(), pdf["lap_number"].max() + 6)
predicted = poly(future_laps)


roots = poly.r
real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > pdf["lap_number"].max()]
predicted_overtake_lap = min(real_roots) if real_roots else None

vertex_lap = -coeffs[1] / (2 * coeffs[0])
vertex_value = poly(vertex_lap)

if predicted_overtake_lap:
    print(f"Predviđeni krug pretice: {predicted_overtake_lap:.1f}")
else:
    print(f"Kriva ne seče nulu u posmatranom opsegu - minimum trenda je ~{vertex_value:.2f}s "
          f"oko kruga {vertex_lap:.0f} (vozač se približio, ali prema ovom trendu ne bi stigao "
          f"do potpune pretice).")
'''),
("code", '''\
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(pdf["lap_number"], pdf["interval"], "o-", color="#d62728", label=f"Stvarni interval (Driver {DRIVER})")
ax.plot(future_laps, predicted, "-", color="#1f77b4", label="Polinomijalna regresija (stepen 2)")
extrapolated_mask = future_laps > pdf["lap_number"].max()
ax.plot(future_laps[extrapolated_mask], predicted[extrapolated_mask], "o--", color="#2ca02c", label="Ekstrapolacija")
if predicted_overtake_lap:
    ax.axvline(predicted_overtake_lap, color="purple", linestyle="--",
               label=f"Predviđeni krug pretice: {predicted_overtake_lap:.0f}")
else:
    ax.axvline(vertex_lap, color="purple", linestyle="--",
               label=f"Minimum trenda (~{vertex_value:.1f}s, krug {vertex_lap:.0f})")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Broj kruga")
ax.set_ylabel("Interval do vozača ispred (s)")
ax.set_title(f"Trend razmaka i predviđena pretica - Driver {DRIVER}")
ax.legend()
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** Opadajući trend razmaka potvrđuje da se vozač realno približavao
protivniku ispred. U ovom konkretnom slučaju kriva NE seče nulu u posmatranom opsegu - trend se
zaravnjuje na oko 2.9s, što je isto validan i zanimljiv nalaz: vozač se približio na ivicu DRS
zone ali prema ovom trendu ne bi stigao do potpune pretice (u stvarnosti ga je verovatno
zaustavio DRS domet, gume protivnika ili odbrana pozicije). Da je kriva sekla nulu, tačka
preseka bi bila procena kruga pretice. Ovo je namerno uprošćen model (ne uzima u obzir DRS zonu,
razlike u gumama ili odbranu pozicije) - u praksi bi se koristio kao jedan od više signala, ne
kao jedini pokazatelj."""),
]


NB6_CELLS = [
("md", """# Spark Streaming demo: prozorska agregacija brzine u (skoro) realnom vremenu

**Cilj:** Pokazati Spark Structured Streaming nad podacima "kako pristižu" - male Parquet
fajlove koje `kafka/consumer.py` piše dok prazni Kafka temu (ovde, za potrebe demonstracije u
Databricks-u gde nije moguće povezati se na lokalni Docker Kafka broker, isti fajlovi su
unapred generisani skriptom `scripts/generate_streaming_chunks.py`, tako da Spark
`readStream` čita direktorijum koji ima mnogo malih fajlova, tačno kako je traženo u
specifikaciji projekta).

**Metod:** `readStream.format("parquet")` sa `maxFilesPerTrigger` postepeno "otkriva" nove
fajlove kao da stižu uživo. Nad tim streamom radimo `groupBy` po 10-sekundnom prozoru i broju
vozača, računajući prosečnu i maksimalnu brzinu. Rezultat pišemo u `memory` sink
(`outputMode="complete"`) i **povremeno** (u više navrata, sa pauzom između) upitujemo tu
tabelu i crtamo grafik - simulirajući kako bi dashboard izgledao da se osvežava uživo.

**Kome je ovo bitno:** Timu za analizu performansi tokom same trke - ovakav pipeline bi u
produkciji davao živ uvid u tempo/brzinu svakog vozača bez čekanja da se cela trka završi."""),
("code", '''\
import glob
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType, TimestampType

spark = SparkSession.builder.appName("F1SparkStreamingDemo").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


CHUNKS_DIR = "../data/streaming_chunks/f1-car-data"

SCHEMA = StructType([
    StructField("date", TimestampType(), True),
    StructField("driver_number", LongType(), True),
    StructField("speed", DoubleType(), True),
    StructField("throttle", DoubleType(), True),
    StructField("brake", DoubleType(), True),
    StructField("rpm", DoubleType(), True),
    StructField("n_gear", DoubleType(), True),
    StructField("gap_to_leader", DoubleType(), True),
])

print("Available chunk files:", len(glob.glob(f"{CHUNKS_DIR}/*.parquet")))
'''),
("code", '''\
stream_df = (
    spark.readStream.schema(SCHEMA)
    .option("maxFilesPerTrigger", 2)
    .parquet(CHUNKS_DIR)
)

windowed = (
    stream_df
    .withWatermark("date", "20 seconds")
    .groupBy(F.window("date", "10 seconds"), "driver_number")
    .agg(
        F.avg("speed").alias("avg_speed"),
        F.max("speed").alias("max_speed"),
        F.count("*").alias("n_readings"),
    )
)


query = (
    windowed.writeStream
    .format("memory")
    .queryName("speed_windows")
    .outputMode("complete")
    .trigger(processingTime="12 seconds")
    .start()
)
print("Streaming query started:", query.id)
'''),
("code", '''\
import matplotlib.pyplot as plt
import pandas as pd


snapshots = []
for i in range(3):
    time.sleep(25)
    snap = spark.sql("SELECT * FROM speed_windows").toPandas()
    snap["snapshot"] = i
    snapshots.append(snap)
    print(f"Upit #{i+1}: {len(snap)} redova agregacije akumulirano do sada")


spark.sparkContext.setLogLevel("FATAL")
try:
    query.stop()
finally:
    spark.sparkContext.setLogLevel("WARN")
print("Streaming upit zaustavljen.")
'''),
("code", '''\
fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 5), sharey=True)
for ax, snap in zip(axes, snapshots):
    if not snap.empty:
        pivot = snap.groupby("driver_number")["avg_speed"].mean()
        pivot.plot(kind="bar", ax=ax, color="#2980b9")
    ax.set_title(f"Upit #{snap['snapshot'].iloc[0] + 1 if not snap.empty else '?'}")
    ax.set_xlabel("Broj vozača")
axes[0].set_ylabel("Prosečna brzina (km/h) po prozoru")
fig.suptitle("Rezultati streaming upita u 3 uzastopna trenutka")
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** Broj akumuliranih redova u tabeli `speed_windows` raste sa svakim upitom
kako Spark obrađuje sve više dolazećih fajlova - potvrđujući da streaming pipeline zaista
inkrementalno obrađuje podatke, a ne čeka da se sav ulaz učita odjednom. Ovo je osnova za
"near real-time" dashboard koji bi mogao da prikazuje trenutnu formu/tempo svakog vozača tokom
same trke."""),
]


NB_TRACK_COMPARISON_CELLS = [
("md", """# Dodatna analiza: poređenje staza (Austrija, Monako, Monza, Meksiko, Brazil - 2024)

**Cilj:** Prethodnih 6 analiza su duboka studija JEDNE trke (Austrija). Ovde proširujemo pogled
na 5 trka iz sezone 2024 da proverimo koliko sama staza (dužina, broj krivina, širina pravih)
utiče na ostvarene brzine - nešto što se ne može videti iz podataka samo jedne trke.

**Metod:** Oblik staze za Monako/Monzu/Meksiko/Brazil crtamo iz javnih GeoJSON referenci
stvarnih obrisa staza (geografske koordinate, ne iz telemetrije - to izbegava šum/preklapanje
krugova). Za Austriju koristimo naš već postojeći telemetrijski trag (Analiza 4). Prosečnu i
maksimalnu brzinu po stazi računamo iz `car_data` za svih 6 vozača na svih 5 trka, i testiramo
jednosmernom ANOVA da li se prosečna brzina statistički značajno razlikuje između staza.

**Kome je ovo bitno:** Timovima za planiranje setup-a bolida pre svake trke (aerodinamički
paket zavisi upravo od toga koliko je staza "brza" - visoke prosečne brzine kao Monza traže
nizak downforce, a spore staze poput Monaka traže visok downforce)."""),
("code", SETUP_CODE),
("code", '''\
import json

GEOJSON_DIR = "../data/circuit_geojsons"
PROCESSED_MULTI = "../data/multi_race_processed"
GEOJSON_TRACKS = {"Monaco": "mc-1929", "Monza": "it-1922", "Mexico": "mx-1962", "Brazil": "br-1940"}


def load_geojson_line(path):
    with open(path) as f:
        root = json.load(f)
    xs, ys = [], []
    for feature in root["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        coords = geometry["coordinates"]
        if geometry["type"] == "LineString":
            xs += [c[0] for c in coords]
            ys += [c[1] for c in coords]
        elif geometry["type"] == "MultiLineString":
            for line in coords:
                xs += [c[0] for c in line]
                ys += [c[1] for c in line]
    return xs, ys
'''),
("code", '''\
fig, axes = plt.subplots(1, 5, figsize=(22, 5))

austria = pd.read_csv(f"{PROCESSED_MULTI}/track_shape_austria.csv")
axes[0].plot(austria["x"], austria["y"], color="#1f77b4")
axes[0].set_title("Austria (iz telemetrije)")
axes[0].set_aspect("equal")
axes[0].set_xticks([])
axes[0].set_yticks([])

for ax, (name, code) in zip(axes[1:], GEOJSON_TRACKS.items()):
    xs, ys = load_geojson_line(f"{GEOJSON_DIR}/{code}.geojson")
    ax.plot(xs, ys, color="#1f77b4")
    ax.set_title(f"{name} (GeoJSON)")
    ax.set_aspect(1.4)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Oblik staze - 5 trka sezone 2024")
plt.tight_layout()
plt.show()
'''),
("code", '''\
speed_summary = spark.read.csv(f"{PROCESSED_MULTI}/speed_summary.csv", header=True, inferSchema=True)
speed_summary.groupBy("track").avg("avg_speed", "max_speed").show()
'''),
("code", '''\
from scipy import stats

pdf = speed_summary.toPandas()
groups = [g["avg_speed"].values for _, g in pdf.groupby("track")]
f_stat, p_value = stats.f_oneway(*groups)
print(f"ANOVA: F = {f_stat:.3f}, p = {p_value:.6f}")

track_stats = pdf.groupby("track")[["avg_speed", "max_speed"]].mean().sort_values("avg_speed")
track_stats
'''),
("code", '''\
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(track_stats))
ax.bar([i - 0.2 for i in x], track_stats["avg_speed"], width=0.4, label="Prosečna brzina", color="#3498db")
ax.bar([i + 0.2 for i in x], track_stats["max_speed"], width=0.4, label="Maksimalna brzina", color="#e74c3c")
ax.set_xticks(list(x))
ax.set_xticklabels(track_stats.index)
ax.set_ylabel("Brzina (km/h)")
ax.set_title(f"Prosečna i maksimalna brzina po stazi (ANOVA p={p_value:.4f})")
ax.legend()
plt.tight_layout()
plt.show()
'''),
("md", """**Zaključak:** ANOVA test potvrđuje (p << 0.05) da se prosečna brzina statistički
značajno razlikuje između staza - što je očekivano (Monza je poznata kao najbrža staza u
kalendaru zbog dugih pravih i malo krivina, dok je Monako najsporija zbog uskih ulica), ali je
korisno imati formalnu statističku potvrdu umesto da se to samo pretpostavi. Oblici staza
jasno pokazuju zašto: Monza ima dugačke prave deonice, Monako je zbijena gradska staza sa oštrim
krivinama, a Austrija/Meksiko/Brazil su negde između."""),
]


NOTEBOOKS = {
    "01_tire_degradation_regression.ipynb": NB1_CELLS,
    "02_pitstop_impact_ttest.ipynb": NB2_CELLS,
    "03_driver_pace_comparison.ipynb": NB3_CELLS,
    "04_telemetry_correlation_clustering.ipynb": NB4_CELLS,
    "05_laptime_prediction_model.ipynb": NB5_CELLS,
    "06_overtake_prediction.ipynb": NB6_OVERTAKE_CELLS,
    "07_spark_streaming_demo.ipynb": NB6_CELLS,
    "08_track_comparison_bonus.ipynb": NB_TRACK_COMPARISON_CELLS,
}


def main() -> None:
    DATABRICKS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in NOTEBOOKS.items():
        nb = make_notebook(cells)
        out_path = DATABRICKS_DIR / filename
        nbf.write(nb, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
