import torch
import torch.nn as nn

"""
Blok rezydualny ma na zadaniu pomóc modelowi wejść na wyższy poziom wiedzy, model przestaje 
szukać pojedyńczych kresek zapisanych na tensorze obrazu lecz złożonych informacji zapisanych 
w nim, robi to po przez łączenie wniosków które zostały wyciągnięte z poprzedniej warswty 
konwolucyjnej. Następnie po tej operacji znajduje się połącznie rezydualne które ma na celu w 
moim rozumieniu spełnić dwa główne założenia: 

Pierwsze - Czyli to żeby syganł się nie popsuł, jeżeli np.: sieć dostanie złe wagi 
           na początku treningu to może to całkowicie wymazać i zniszczyć informacje o obrazie, 
           dzięki połączeniu rezydualnemu Dzięki połączeniu rezidualnemu, najgorsze co model może 
           zrobić, to ustawić wszystkie wagi splotów na zero. Nawet jeśli tak się stanie i sploty 
           wyplują kompletną pustkę (out = 0), to dzięki operacji 'out += residual' sieć i tak 
           przekaże dalej nienaruszony, oryginalny sygnał 'x'. 

Drugie - Czyli zabezpieczenie przed zanikiem informacji i sygnału inaczej również znane jako 
         "zanikanie gradientu". Podczas wstecznej propagacji (gdy model się uczy i poprawia swoje 
         błędy), sygnał z błędem (gradient) wraca od końca sieci do początku. Przechodząc 
         "wstecz" przez zwykłe warstwy splotowe, gradient jest wielokrotnie mnożony przez ułamki. 
         Im sieć jest głębsza, tym ten sygnał staje się mniejszy, aż na samym początku 
         sieci (w pierwszych warstwach) wynosi dokładnie zero efektem jest że pierwsze warstwy 
         przestają się uczyć. Połączenie rezidualne rozwiązuje to za pomocą zwykłego matematycznego 
         dodawania (+). W matematyce pochodna z dodawania działa tak, że przepuszcza sygnał bez żadnych 
         zmian. Dodatkowo Połączenie skrótowe tworzy dla gradientu "autostradę bez opłat". Sygnał o 
         błędzie może swobodnie, w nienaruszonej formie, przeskoczyć obok splotów i dotrzeć prosto do 
         wcześniejszych warstw sieci.

W funkcji forward robimy połączenie rezydualne odpiero po drugiej warstwie konwolucyjnej oraz 
przed ostatnią w bloku warstwą LeakyReLU, wklejamy naszą kopię cech z początku dokładnie wtedy 
ponieważ wcześniej by się nam nie opłacało zakłucili byśmy wtedy pracę warstw konwolucyjnych oraz 
popsuli byśmy łączenie cech w bardziej zwięzłe i skomplikowane konstrukty, robimy to również przed 
warstwa aktywacyjną ponieważ chcemy połączyć pełny profil matematyczny. Kiedy robimy 
'out += residual' łączymy pełną informację z obu ścieżek. Dopiero gdy ta fuzja się dokona, 
nakładamy na sam koniec self.relu(out). LeakyReLU działa jak ostateczny cenzor – patrzy na wynik 
tego połączenia i mówi: "Okej, po zsumowaniu wszystkiego te wartości są silne i ważne, więc je 
przepuszczam dalej, a tamte okazały się szumem, więc je wygaszam".  
"""
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class PlateLocNet(nn.Module):
    def __init__(self):
        super(PlateLocNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )

        self.deep_features = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        self.confidence_branch = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 1, kernel_size=1)
        )

        self.box_branch = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 4, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.deep_features(x)

        confidence_out = self.confidence_branch(x)
        box_out = self.box_branch(x)
        out = torch.cat([confidence_out, box_out], dim=1)
        out = out.permute(0, 2, 3, 1)

        return torch.sigmoid(out)