# BMP2WAV Convertor

## Funkce Převodníku

Funkce, které aplikace poskytuje, a technologické schopnosti, které umožňují obousměrný převod mezi obrazovými a zvukovými daty, jejich generování a vizualizaci.

### Generace Náhodného Obrázku

Aplikace umožňuje generovat náhodné rastrové obrázky ve formátu BMP s definovanou šířkou a výškou.

#### Vlastnosti
- Rozlišení obrázku je zadáváno uživatelem pomocí šířky a výšky v pixelech.
- Barevné složky (R, G, B) jsou generovány nezávisle pro každý pixel.
- Formát výstupu je 24bitový nekomprimovaný BMP.
- Použití: Výstupní obrázek může sloužit jako vstup pro převod na WAV nebo jako testovací soubor.

#### Parametry Příkazové Řádky
```bash
py ./main.py -gb <sirka> <vyska> <vystupni_cesta> [--show -s] [--show-with-axes -sx] [--test]
```

#### Volitelné Přepínače
- `-s`: Zobrazí vygenerovaný obrázek.
- `-sx`: Zobrazí obrázek s osami.
- `--test`: Uloží zobrazení do testovací složky.

#### Příklad Použití
```bash
py ./main.py -gb 100 100 ./content/media/out/random.bmp -s
```

---

### Generace Náhodného Zvuku

Aplikace umožňuje generovat náhodný zvukový signál ve formátu WAV.

#### Vlastnosti
- Délka zvuku je zadána uživatelem v počtu vzorků.
- Vzorkovací frekvence: 44 100 Hz.
- Počet kanálů: 1 (mono).
- Bitová hloubka: 32 bitů na vzorek.
- Formát výstupu: WAV PCM (nekomprimovaný).

#### Parametry Příkazové Řádky
```bash
py ./main.py -gw <pocet_vzorku> <vystupni_cesta> [--show -s] [--show-with-axes -sx] [--test]
```

#### Volitelné Přepínače
- `-s`: Zobrazí zvukovou vlnu.
- `-sx`: Zobrazí zvukovou vlnu s osami.
- `--test`: Uloží zobrazení do testovací složky.

#### Příklad Použití
```bash
py ./main.py -gw 1000 ./content/media/out/random_wave.wav -s
```

---

### Generace Sinusového Signálu

Aplikace umožňuje vytvořit čistý sinusový signál ve formátu WAV.

#### Vlastnosti
- Frekvence signálu: Uživatelsky definovatelná.
- Vzorkovací frekvence: 44 100 Hz.
- Počet vzorků: Uživatelsky definovatelný.
- Počet kanálů: 1 (mono).
- Bitová hloubka: 32 bitů na vzorek.
- Formát výstupu: WAV PCM (nekomprimovaný).

#### Parametry Příkazové Řádky
```bash
py ./main.py -gw <pocet_vzorku> <vystupni_cesta> -f <frekvence> [--show -s] [--show-with-axes -sx] [--test]
```

#### Příklad Použití
```bash
py ./main.py -gw 1000 ./content/media/out/sine_wave.wav -f 440 -s
```

---

### Převod BMP na WAV

Aplikace umožňuje převod rastrového obrázku ve formátu BMP na zvukový soubor WAV.

#### Vlastnosti
- Podporované BMP formáty: 1bit, 4bit, 8bit, 24bit.
- Výstupní WAV:
   - Vzorkovací frekvence: 44 100 Hz.
   - Hloubka: 32 bitů.
   - Mono.

#### Parametry Příkazové Řádky
```bash
python main.py --convert <vstup.bmp> <vystup.wav> [--show -s] [--show-with-axes -sx] [--test]
```

#### Příklad Použití
```bash
python main.py --convert ./content/media/bmp/1bit.bmp ./content/media/out/1bit.wav --show
```

---

### Převod WAV na BMP

Aplikace umožňuje převést zvukový soubor ve formátu WAV zpět na rastrový obrázek BMP.

#### Vlastnosti
- Převod amplitud zpět na RGB hodnoty.
- Rekonstrukce původního rozměru obrázku.

#### Parametry Příkazové Řádky
```bash
python main.py --convert <vstup.wav> <vystup.bmp> [-c width height] [--show -s] [--show-with-axes -sx] [--test]
```

#### Příklad Použití
```bash
py ./main.py --convert ./content/media/out/1bit.wav ./content/media/out/1bit.bmp -s
```

---

### Zobrazování Dat Souborů

Aplikace umožňuje vizualizaci dat z WAV a BMP souborů.

#### Parametry Příkazové Řádky
```bash
py ./main.py -i <soubor.wav nebo soubor.bmp> [--show -s] [--show-with-axes -sx] [--test]
```

#### Příklad Použití
```bash
py ./main.py -i ./content/media/bmp/24bit.bmp -s
```
