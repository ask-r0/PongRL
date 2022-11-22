# Reinforcement Learning: Atari 2600 Pong
Prosjekt utført i faget IDATT2502 utført høsten 2022.

Github-mappen  består av en implementasjon av Deep Q Learning algoritme ved bruk av Deep-Q-Networks.

## Kjøre programmet
Kjør programmet ved følgende kommandoer:

```
python3 -m venv env
source env/bin/activate
pip3 install -r req.txt
python3 src/main.py
```

_Kommandoer kun testet på macOS._

## Funksjonalitet
### Se trent nettverk spille
Nettverk som lastes må være trent på GPU. Foreløpig er kun miljøet Pong støttet.

`python3 main.py play <network-path> <network-type> <num-frames>`

* `<network-path>` er stien til nettverket
* `<network-type>` er typen nettverk. Enten "cnn" eller "dueling".
* `<num-frames>` er antall steg før programmet avslutter

### Trene nettverk fra settings (instillinger)
`python3 main.py train <settings-path>`

Se settings-mappen for all informasjon om hvordan settings skal formateres.

## Demovideo
Følgende video viser utviklingen til en agent under trening:

https://youtu.be/XtnDACZDNX4

_Videoen viser bare trening fram til 250k frames, test programmet selv med det ferdige trente nettet example.pth 
ved kommandoen: _ `python3 src/main.py play pre_trained/example.pth cnn 1000`

