"""
Configuration des styles et constantes de l'interface utilisateur.
"""

# Couleurs
PRIMARY_COLOR = '#3B82F6'  # Bleu
SUCCESS_COLOR = '#10B981'  # Vert
ERROR_COLOR = '#EF4444'    # Rouge
WARNING_COLOR = '#F59E0B'  # Orange
INFO_COLOR = '#3B82F6'     # Bleu

# Styles des cartes
CARD_SHADOW = 'shadow-lg'
CARD_RADIUS = 'rounded-xl'
CARD_BG = 'bg-slate-800'
CARD_TEXT = 'text-white'

# Styles des boutons
BUTTON_PRIMARY = f'bg-{PRIMARY_COLOR} text-white px-4 py-2 rounded-lg hover:opacity-90'
BUTTON_SUCCESS = f'bg-{SUCCESS_COLOR} text-white px-4 py-2 rounded-lg hover:opacity-90'
BUTTON_ERROR = f'bg-{ERROR_COLOR} text-white px-4 py-2 rounded-lg hover:opacity-90'
BUTTON_WARNING = f'bg-{WARNING_COLOR} text-white px-4 py-2 rounded-lg hover:opacity-90'

# Styles des tableaux
TABLE_HEADER = 'bg-slate-700 text-white font-semibold'
TABLE_ROW = 'hover:bg-slate-700'
TABLE_CELL = 'px-4 py-2'

# Styles des graphiques
CHART_BG = 'bg-slate-800'
CHART_TEXT = 'text-white'
CHART_GRID = 'rgba(255, 255, 255, 0.1)'

# Styles des inputs
INPUT_BG = 'bg-slate-700'
INPUT_TEXT = 'text-white'
INPUT_BORDER = 'border-slate-600'
INPUT_FOCUS = 'focus:border-blue-500 focus:ring-1 focus:ring-blue-500'

# Styles des labels
LABEL_TEXT = 'text-white font-semibold'
LABEL_SMALL = 'text-sm text-slate-400'

# Styles des badges
BADGE_SUCCESS = f'bg-{SUCCESS_COLOR} text-white px-2 py-1 rounded-full text-xs'
BADGE_ERROR = f'bg-{ERROR_COLOR} text-white px-2 py-1 rounded-full text-xs'
BADGE_WARNING = f'bg-{WARNING_COLOR} text-white px-2 py-1 rounded-full text-xs'
BADGE_INFO = f'bg-{INFO_COLOR} text-white px-2 py-1 rounded-full text-xs'

# Styles des icônes
ICON_SMALL = 'text-lg'
ICON_MEDIUM = 'text-2xl'
ICON_LARGE = 'text-3xl'

# Styles des titres
TITLE_LARGE = 'text-3xl font-bold text-white'
TITLE_MEDIUM = 'text-2xl font-bold text-white'
TITLE_SMALL = 'text-xl font-bold text-white'

# Styles des conteneurs
CONTAINER_PADDING = 'p-4'
CONTAINER_MARGIN = 'm-4'
CONTAINER_GAP = 'gap-4'

# Styles des grilles
GRID_2_COLS = 'grid grid-cols-2 gap-4'
GRID_3_COLS = 'grid grid-cols-3 gap-4'
GRID_4_COLS = 'grid grid-cols-4 gap-4'

# Styles des flexbox
FLEX_ROW = 'flex flex-row gap-4'
FLEX_COL = 'flex flex-col gap-4'
FLEX_CENTER = 'flex items-center justify-center'
FLEX_BETWEEN = 'flex items-center justify-between'
FLEX_AROUND = 'flex items-center justify-around'

# Styles des animations
ANIMATION_FADE = 'transition-opacity duration-300'
ANIMATION_SLIDE = 'transition-transform duration-300'
ANIMATION_SCALE = 'transition-transform duration-300'

# Styles des tooltips
TOOLTIP_BG = 'bg-slate-900'
TOOLTIP_TEXT = 'text-white'
TOOLTIP_PADDING = 'px-2 py-1'
TOOLTIP_RADIUS = 'rounded'

# Styles des modales
MODAL_BG = 'bg-slate-900'
MODAL_TEXT = 'text-white'
MODAL_PADDING = 'p-6'
MODAL_RADIUS = 'rounded-xl'
MODAL_SHADOW = 'shadow-2xl'

# Styles des onglets
TAB_ACTIVE = 'bg-slate-700 text-white'
TAB_INACTIVE = 'text-slate-400 hover:text-white'
TAB_PADDING = 'px-4 py-2'
TAB_BORDER = 'border-b-2 border-transparent'
TAB_BORDER_ACTIVE = 'border-blue-500' 