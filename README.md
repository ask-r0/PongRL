# Reinforcement Learning: Atari 2600 Pong
Prosjekt utført i faget IDATT2502 utført høsten 2022.

Github-mappen  består av en implementasjon av Deep Q Learning algoritme ved bruk av Deep-Q-Networks.

## Dependencies
Følgende biblioteker må installeres for å kjøre programmet:
* Gym (https://www.gymlibrary.dev/). Kommando: `pip3 install "gym[atari,accept-rom-license]==0.21.0"`
* PyTorch (https://pytorch.org/). Kommando: `pip3 install torch`

_Kommandoer kun testes på macOS og Google Colab._

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

