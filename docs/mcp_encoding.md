# Encodage Match Charting Project utilisé en Phase 9

## Source et portée

La référence primaire est l'onglet `Instructions` du tableur officiel MatchChart 0.3.2, recoupé avec le guide officiel Tennis Abstract et les chaînes réellement présentes dans l'instantané local. Le MCP est distribué sous licence CC BY-NC-SA 4.0. L'instantané local ne contient que les fichiers hommes (`charting-m-*`); aucune donnée WTA n'est inventée ou téléchargée implicitement.

## Grammaire

Une ligne représente un point. `1st` contient le premier service et, s'il est bon, tout l'échange. En cas de faute, `1st` contient seulement cette tentative et `2nd` contient le second service puis l'échange.

### Service

- Direction: `4` wide, `5` corps, `6` T, `0` inconnue.
- `c`: let, répétable avant le service.
- `+`: tentative de service-volée.
- `*`: ace; `#`: service non retourné.
- Faute: `n` filet, `w` large, `d` longue, `x` large et longue, `g` faute de pied, `e` type inconnu, `!` bois/shank.
- Le modificateur `;` est conservé comme net cord, y compris dans sa variante historique placée après la faute.
- `^` après un service existe dans l'instantané récent mais sa sémantique n'est pas définie par le tableur officiel disponible. La frontière du token est parsée, la chaîne brute est conservée et `has_explicit_unknown=true`; aucune signification tactique ne lui est attribuée.

### Coups

| Code | Type |
|---|---|
| `f`, `b` | coup droit, revers |
| `r`, `s` | slice/chip coup droit, revers |
| `v`, `z` | volée coup droit, revers |
| `o`, `p` | smash standard, smash revers |
| `u`, `y` | amortie coup droit, revers |
| `l`, `m` | lob coup droit, revers |
| `h`, `i` | demi-volée coup droit, revers |
| `j`, `k` | volée liftée coup droit, revers |
| `t` | trick shot |
| `q` | type inconnu explicite |

Direction: `1` vers le côté coup droit d'un droitier/côté revers d'un gaucher, `2` milieu, `3` côté revers d'un droitier/côté coup droit d'un gaucher, `0` inconnue. La relation crosscourt/long de ligne est dérivée avec la main du frappeur; la zone source brute reste conservée.

Profondeur: `7` zone de service, `8` entre ligne de service et milieu du fond, `9` proche du fond, `0` inconnue. La documentation historique la réserve au retour, mais l'instantané 2020–2026 l'emploie sans ambiguïté après de nombreux coups de rallye (`f39`, `b28`, etc.). Le parseur accepte donc cette extension sur tous les coups et la documente comme variante observée.

Position/modificateurs: `+` approche, `-` frappe au filet, `=` frappe au fond, `;` net cord, `^` stop/drop volley. Dans les données réelles ces codes peuvent précéder ou suivre direction/profondeur. `c` à l'intérieur d'un rallye est une rare variante observée de net cord et reste signalée par la chaîne brute.

Fin: `*` winner, `@` faute directe, `#` faute provoquée. Les types de faute utilisent les mêmes codes que le service. Les variantes réelles telles que `b!1w#` sont conservées comme shank + direction + faute large provoquée.

### Points spéciaux

`P` pénalité contre le serveur, `Q` pénalité contre le relanceur, `R` point non observé gagné par le relanceur, `S` point non observé gagné par le serveur. Ils sont valides pour le score mais exclus du dataset shot-level car aucune frappe n'est observable. `C` (arrêt sur challenge incorrect) est également exclu des séquences tactiques.

## Politique de sûreté

Le parseur est strict: syntaxe inconnue, fin absente, caractères après une fin de point, métadonnée critique invalide ou vainqueur reconstruit différent de `PtWinner` entraînent une exclusion avec motif et exemples dans `reports/mcp_parser_audit.md`. Les inconnues prévues (`0`, `q`, `e`) sont des catégories explicites. Aucun code inconnu n'est silencieusement converti en coup connu.

## Exemples tests

- `4ffbbf*`: service wide, cinq frappes, winner final.
- `5d` / `4s39b3b1w@`: premier service long, second service wide, retour slice direction 3 profond, échange, faute directe large.
- `4+s28v1f-3*`: service-volée, retour slice, volée, winner de coup droit frappé au filet.
- `4n` / `5d`: double faute.
