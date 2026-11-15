"""Benchmark selection: prose authors of the 3rd-early 5th c. AH.

Each entry: OpenITI book URI -> (genre, status).
  status 'secure'   = attribution not seriously contested; used for training/testing
  status 'disputed' = attribution contested; held out as a test case
  status 'single'   = author's only secure prose work in OpenITI; impostor pool only

Genres (coarse): adab, history, criticism (naqd/balagha/lexicology),
ethics (akhlaq/philosophy), zuhd (isnad-heavy piety collections), law.
"""

BOOKS = {
    # al-Jahiz (d. 255)
    "0255Jahiz.BayanWaTabyin": ("adab", "secure"),
    "0255Jahiz.Hayawan": ("adab", "secure"),
    "0255Jahiz.Bukhala": ("adab", "secure"),
    "0255Jahiz.Rasail": ("adab", "secure"),
    "0255Jahiz.BursanWaCurjan": ("adab", "secure"),
    "0255Jahiz.Cuthmaniyya": ("adab", "secure"),
    "0255Jahiz.TarbicWaTadwir": ("adab", "secure"),
    "0255Jahiz.MahasinWaAddad": ("adab", "disputed"),
    "0255Jahiz.TajFiAkhlaq": ("adab", "disputed"),
    # Ibn Qutayba (d. 276)
    "0276IbnQutaybaDinawari.CuyunAkhbar": ("adab", "secure"),
    "0276IbnQutaybaDinawari.Macarif": ("history", "secure"),
    "0276IbnQutaybaDinawari.ShicrWaShucara": ("criticism", "secure"),
    "0276IbnQutaybaDinawari.AdabKatib": ("criticism", "secure"),
    "0276IbnQutaybaDinawari.TawilMukhtalafHadith": ("law", "secure"),
    "0276IbnQutaybaDinawari.TawilMushkilQuran": ("criticism", "secure"),
    "0276IbnQutaybaDinawari.ImamaWaSiyasa": ("history", "disputed"),
    # Baladhuri (d. 279)
    "0279Baladhuri.FutuhBuldan": ("history", "secure"),
    "0279Baladhuri.AnsabAshraf": ("history", "secure"),
    # Ibn Abi l-Dunya (d. 281): isnad-heavy, a stress test for quotation removal
    "0281IbnAbiDunya.Zuhd": ("zuhd", "secure"),
    "0281IbnAbiDunya.DhammDunya": ("zuhd", "secure"),
    "0281IbnAbiDunya.Samt": ("zuhd", "secure"),
    "0281IbnAbiDunya.MakarimAkhlaq": ("zuhd", "secure"),
    # Mubarrad (d. 285)
    "0285IbnYazidMubarrad.KamilFiLugha": ("adab", "secure"),
    "0285IbnYazidMubarrad.Tacazi": ("adab", "secure"),
    "0285IbnYazidMubarrad.Fadil": ("adab", "secure"),
    # Yacqubi (d. 292)
    "0292Yacqubi.Tarikh": ("history", "secure"),
    "0292Yacqubi.Buldan": ("history", "secure"),
    # Ibn al-Muctazz (d. 296)
    "0296IbnMuctazz.TabaqatShucara": ("criticism", "secure"),
    "0296IbnMuctazz.Badic": ("criticism", "secure"),
    # Tabari (d. 310)
    "0310Tabari.Tarikh": ("history", "secure"),
    "0310Tabari.IkhtilafFuqaha": ("law", "secure"),
    "0310Tabari.MuntakhabMinDhayl": ("history", "secure"),
    # Suli (d. 335)
    "0335Suli.AdabKuttab": ("criticism", "secure"),
    "0335Suli.AkhbarRadi": ("history", "secure"),
    "0335Suli.KhilafatWathiq": ("history", "secure"),
    "0335Suli.AkhbarAbiTammam": ("criticism", "secure"),
    "0335Suli.AkhbarShucara": ("criticism", "secure"),
    # Qudama b. Jacfar (d. 337)
    "0337QudamaIbnJacfar.Kharaj": ("history", "secure"),
    "0337QudamaIbnJacfar.NaqdShicr": ("criticism", "secure"),
    # Mascudi (d. 346)
    "0346Mascudi.MurujDhahab": ("history", "secure"),
    "0346Mascudi.TanbihWaIshraf": ("history", "secure"),
    # Hamza al-Isfahani (d. 360)
    "0360HamzaIsfahani.DurraFakhira": ("adab", "secure"),
    "0360HamzaIsfahani.TarikhSini": ("history", "secure"),
    "0360HamzaIsfahani.Tanbih": ("criticism", "secure"),
    # Abu l-Faraj al-Isfahani (d. 356; OpenITI dates him 362)
    "0362AbuFarajIsbahani.Aghani": ("adab", "secure"),
    "0362AbuFarajIsbahani.MaqatilTalibiyyin": ("history", "secure"),
    "0362AbuFarajIsbahani.Diyarat": ("adab", "secure"),
    "0362AbuFarajIsbahani.ImaShawacir": ("adab", "secure"),
    "0362AbuFarajIsbahani.AdabGhuraba": ("adab", "disputed"),
    # Marzubani (d. 384)
    "0384IbnCimranMarzubani.Muwashshah": ("criticism", "secure"),
    "0384IbnCimranMarzubani.MucjamShucara": ("criticism", "secure"),
    # al-Muhassin al-Tanukhi (d. 384)
    "0384MuhassinTanukhi.FarajBacdaShidda": ("adab", "secure"),
    "0384MuhassinTanukhi.NishwarMuhadara": ("adab", "secure"),
    "0384MuhassinTanukhi.Mustajad": ("adab", "secure"),
    # Abu Hilal al-Caskari (d. 395)
    "0395AbuHilalCaskari.Sinacatayn": ("criticism", "secure"),
    "0395AbuHilalCaskari.Awail": ("history", "secure"),
    "0395AbuHilalCaskari.FuruqLughawiyya": ("criticism", "secure"),
    "0395AbuHilalCaskari.DiwanMacani": ("adab", "secure"),
    "0395AbuHilalCaskari.JamharatAmthal": ("adab", "secure"),
    # Abu Hayyan al-Tawhidi (d. 414)
    "0414AbuHayyanTawhidi.Imtac": ("adab", "secure"),
    "0414AbuHayyanTawhidi.Basair": ("adab", "secure"),
    "0414AbuHayyanTawhidi.AkhlaqWazirayn": ("adab", "secure"),
    "0414AbuHayyanTawhidi.Muqabasat": ("ethics", "secure"),
    "0414AbuHayyanTawhidi.Sadaqa": ("ethics", "secure"),
    # Miskawayh (d. 421)
    "0421Miskawayh.Tajarib": ("history", "secure"),
    "0421Miskawayh.HikmaKhalida": ("ethics", "secure"),
    "0421Miskawayh.HawamilWaShawamil": ("ethics", "secure"),
    "0421Miskawayh.TahdhibAkhlaq": ("ethics", "secure"),
    # al-Thacalibi (d. 429)
    "0429AbuMansurThacalibi.ThimarQulub": ("adab", "secure"),
    "0429AbuMansurThacalibi.FiqhLugha": ("criticism", "secure"),
    "0429AbuMansurThacalibi.Tamthil": ("adab", "secure"),
    "0429AbuMansurThacalibi.SihrBalagha": ("criticism", "secure"),
    "0429AbuMansurThacalibi.LubabAdab": ("adab", "secure"),
    "0429AbuMansurThacalibi.KhassKhass": ("adab", "secure"),
    # Single-work authors: impostor pool
    "0320IbnMuhammadBayhaqi.Mahasin": ("adab", "single"),
    "0328IbnCabdRabbih.CiqdFarid": ("adab", "single"),
    "0331Jahshiyari.Wuzara": ("history", "single"),
    "0282AbuHanifaDinawari.AkhbarTiwal": ("history", "single"),
    "0325IbnIshaqWashsha.Muwashsha": ("adab", "single"),
    "0335IbnWahbKatib.BurhanFiWujuhBayan": ("criticism", "single"),
    "0280IbnTayfur.BallaghatNisa": ("adab", "single"),
}
