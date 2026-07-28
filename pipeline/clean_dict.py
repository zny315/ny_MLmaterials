import re
import argparse
import pandas as pd
from pathlib import Path

try:
    from public_utils import console_print as print, install_ascii_exception_hook
    from schema import canonicalize_dataframe_columns
except ModuleNotFoundError:
    from pipeline.public_utils import console_print as print, install_ascii_exception_hook
    from pipeline.schema import canonicalize_dataframe_columns

install_ascii_exception_hook()

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# Expert name-to-structure dictionary.
# Keys are representative SMILES strings; values are accepted English aliases.
knowledge_base = {
    # Expert dictionary used for discrete molecular or ionic materials.

            # Inorganic flame retardants
            # Aluminum hydroxide (ATH)
            "[OH-].[OH-].[OH-].[Al+3]": [
                "ath", "aluminium hydroxide", "aluminum hydroxide",
                "alumina hydroxide", "alumina hydrate",
                "aluminium trihydrate", "aluminum trihydrate", "alumina trihydrate",
                "aluminium trihydroxide", "aluminum trihydroxide",
                "alumina trihydroxide",
                "al(oh)3", "aluminium hydroxide (ath)", "aluminum hydroxide (ath)",
                "aluminium hydroxide ath", "aluminum hydroxide ath", "aluminium hydroxide (ATH)",
                "aluminium hydroxide trihydrate", "aluminum hydroxide trihydrate",
            ],
            # Magnesium hydroxide (MDH/MH)
            "[OH-].[OH-].[Mg+2]": [
                "mdh", "mh", "magnesium hydroxide", "mg(oh)2", "brucite",
                "magnesium hydroxide (mdh)", "magnesium hydroxide (mh)",
            ],
            # Antimony trioxide (ATO)
            " O1[Sb]2O[Sb]3O[Sb]1O[Sb](O2)O3": ["ato", "antimony trioxide"],
            # Zinc borate (ZB)
            "[Zn+2].[Zn+2].[Zn+2].[O-]B([O-])[O-].[O-]B([O-])[O-]": [
                "zb", "zinc borate", "bzn", "zn6o(oh)(bo3)3",
            ],
            # Zinc hydroxystannate (ZHS)
            "[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[Zn+2].[Sn+4]": ["zhs", "zinc hydroxystannate"],
            # Halogenated flame retardants
            # Tetrabromobisphenol A (TBBPA)
            "CC(C)(C1=CC(=C(C(=C1)Br)O)Br)C2=CC(=C(C(=C2)Br)O)Br": ["tbbpa", "tetrabromobisphenol a"],
            # Decabromodiphenyl ethane (DBDPE)
            "C(CC1=C(C(=C(C(=C1Br)Br)Br)Br)Br)C2=C(C(=C(C(=C2Br)Br)Br)Br)Br": ["dbdpe", "decabromodiphenyl ethane"],
            # Hexabromocyclododecane (HBCD)
            " BrC1C(Br)CCC(Br)C(Br)CCC(Br)C(Br)CC1 ": ["hbcd", "hexabromocyclododecane"],
            # 1,2-bis(2,4,6-tribromophenoxy)ethane (BTBPE)
            "C(COC1=C(C=C(C=C1Br)Br)Br)OC2=C(C=C(C=C2Br)Br)Br": ["btbpe", "1,2-bis(2,4,6-tribromophenoxy)ethane"],
            # Dechlorane Plus
            " C1CC2C(CCC3C1C4(C(=C(C3(C4(Cl)Cl)Cl)Cl)Cl)Cl)C5(C(=C(C2(C5(Cl)Cl)Cl)Cl)Cl)Cl ": ["dp", "ddc-co", "dechlorane plus"],

            # Organophosphorus flame retardants
            # Triphenyl phosphate (TPP/TPhP)
            "C1=CC=C(C=C1)OP(=O)(OC2=CC=CC=C2)OC3=CC=CC=C3": ["tpp", "tphp", "triphenyl phosphate"],
            # Tris(1-chloro-2-propyl) phosphate (TCPP)
            "CC(CCl)OP(=O)(OC(C)CCl)OC(C)CCl": ["tcpp", "tris(1-chloro-2-propyl) phosphate"],
            # Bisphenol A bis(diphenyl phosphate) (BDP)
            "CC(C)(C1=CC=C(C=C1)OP(=O)(OC2=CC=CC=C2)OC3=CC=CC=C3)C4=CC=C(C=C4)OP(=O)(OC5=CC=CC=C5)OC6=CC=CC=C6": ["bdp", "bisphenol-a bis(diphenyl phosphate)"],
            # Resorcinol bis(diphenyl phosphate) (RDP)
            "C1=CC=C(C=C1)OP(=O)(OC2=CC=CC=C2)OC3=CC=CC(=C3)OP(=O)(OC4=CC=CC=C4)OC5=CC=CC=C5 ": ["rdp", "resorcinol bis(diphenyl phosphate)"],
            # DOPO
            "C1=CC2=C(C=C1)P(=O)(OC3=CC=CC=C32)H": [
                "dopo", "9,10-dihydro-9-oxa-10-phosphaphenanthrene-10-oxide",
                "9,10-dihydro-9-oxa-10-phosphaphenanthrene-10-oxidation",
            ],
            # DOPO-HQ
            "C1=CC2=C(C=C1)P(=O)(OC3=CC=CC=C32)C4=C(C=CC(=C4)O)O": ["dopo-hq", "10-(2,5-dihydroxyphenyl)-10h-9-oxa-10-phosphaphenanthrene-10-oxide"],
            # Aluminum diethyl phosphinate (ADP/OP10)
            "CCP(=O)(CC)[O-].CCP(=O)(CC)[O-].CCP(=O)(CC)[O-].[Al+3]": ["adp", "op10", "aluminum diethyl phosphinate", "aluminium diethyl phosphinate"],

            # Nitrogen-containing and intumescent flame retardants
            # Melamine (MEL)
            "C1(=NC(=NC(=N1)N)N)N": ["mel", "melamine"],
            # Melamine cyanurate (MCA)
            "C1(=NC(=NC(=N1)N)N)N.C1(=O)NC(=O)NC(=O)N1": ["mca", "melamine cyanurate"],
            # Ammonium polyphosphate (APP)
            "[NH4+].[NH4+].[NH4+].[O-]P(=O)([O-])[O-]": ["app", "ammonium polyphosphate"],
            # Melamine polyphosphate (MPP)
            "NC1=NC(N)=NC(N)=N1.OP(=O)(O)O": ["mpp", "melamine polyphosphate"],
            # Piperazine pyrophosphate (PAPP/PPAP)
            "C1CNCCN1.[O-]P(=O)(O)OP(=O)(O)[O-]": ["papp", "ppap", "piperazine pyrophosphate"],
            # Pentaerythritol (PER)
            "C(C(CO)(CO)CO)O": ["per", "pentaerythritol"],
            # Dipentaerythritol (DPER/DPE)
            "C(C(CO)(CO)CO)OCC(CO)(CO)CO": [
                "dper", "dpe", "dipentaerythritol", "di-pentaerythritol",
            ],

            # Nanoscale and silicon-based materials
            # Octaisobutyl POSS (Oib-POSS)
            "CC(C)C[Si]12O[Si]3(O[Si]4(O[Si](O1)(O[Si]5(O[Si](O2)(O[Si](O3)(O[Si](O4)(O5)CC(C)C)CC(C)C)CC(C)C)CC(C)C)CC(C)C)CC(C)C)CC(C)C": ["oib-poss", "octaisobutyl-poss"],
            # Octaphenyl POSS (Ph-POSS)
            "C6=CC=CC=C6[Si]12O[Si]3(O[Si]4(O[Si](O1)(O[Si]5(O[Si](O2)(O[Si](O3)(O[Si](O4)(O5)C%13=CC=CC=C%13)C%12=CC=CC=C%12)C%11=CC=CC=C%11)C%10=CC=CC=C%10)C9=CC=CC=C9)C8=CC=CC=C8)C7=CC=CC=C7": ["ph-poss", "octaphenyl-poss"],
            # Titanium carbide MXene
            "[Ti].[C]": ["ti3c2tx", "mxene", "titanium carbide mxene"],
            "O=[Si]=O": [
                "sio2", "silica", "nano-silica", "silica gel", "silica-gel",
                "fumed silica", "silicon dioxide",
                "mesoporous silica", "silica nanoparticles", "silica nanoparticle",
                "snp", "silica gel resin",
            ],
            # Common minerals and inorganic materials
            "[P]": ["rp", "red phosphorus", "red phosphorous", "microencapsulated red phosphorus", "mrp", "mp"],
            "O=C([O-])[O-].[Ca+2]": ["caco3", "calcium carbonate", "chalk"],
            "[Mg+2].[Mg+2].[Mg+2].[Mg+2].[Mg+2].[O-]C(=O)[O-].[O-]C(=O)[O-].[O-]C(=O)[O-].[O-]C(=O)[O-].[OH-].[OH-].[OH2].[OH2].[OH2].[OH2]": [
                "hydromagnesite", "magnesium carbonate hydroxide hydrate",
            ],
            "[Mg+2].[O-2]": ["mgo", "magnesium oxide"],
            "[Ca+2].[Mg+2].[Mg+2].[Mg+2].[O-]C(=O)[O-].[O-]C(=O)[O-].[O-]C(=O)[O-].[O-]C(=O)[O-]": [
                "huntite", "huntite/hydromagnesite mineral",
            ],
            "O=[Si]=O.O=[Si]=O.[Mg+2].[Al+3].[O-2].[O-2].[O-2]": [
                "palygorskite", "attapulgite", "vermiculite", "kaol", "kaolin", "kaolinite",
                "sepiolite",
            ],
            "O=[Si]([O-])[O-].[Na+].[Na+]": [
                "sodium silicate", "water glass",
            ],
            "O=P([O-])([O-])O[Zr]OP(=O)([O-])[O-]": [
                "zrp", "zirconium phosphate", "alpha-zirconium phosphate",
                "\u03b1-zirconium phosphate", "\u00ce\u00b1-zirconium phosphate",
            ],
            "[Ag]": ["silver", "ag"],

            # Polymer-backbone repeat-unit proxies
            # EVA ethylene and vinyl-acetate units
            "C=C.C=COC(C)=O": [
                "eva", "ethylene-vinyl acetate", "ethylene vinyl acetate",
                "ethylene-vinyl acetate copolymer", "ethylene vinyl acetate copolymer",
                "ethylene/vinyl acetate copolymer", "poly(ethylene-co-vinyl acetate)",
                "eva backbone", "ethylene-vinyl acetate backbone", "eva acetate groups",
                "eva copolymer",
            ],
            # Polyethylene repeat unit
            "C=C": [
                "pe", "polyethylene", "polyethylene backbone", "ldpe", "lldpe", "hdpe",
                "uldpe", "vldpe", "low density polyethylene",
                "linear low density polyethylene", "high density polyethylene",
                "ultra low density polyethylene", "very low density polyethylene",
            ],
            "C=C(C)": [
                "polypropylene", "isotactic polypropylene", "syndiotactic polypropylene",
            ],
            # Maleated polyethylene proxy
            "C=C.O=C1OC(=O)C=C1": [
                "pegma", "pe-g-ma", "pe-g-mah", "mpe", "maleated polyethylene",
                "maleated pe", "maleated lldpe pegma", "maleated lldpe pe-g-ma",
                "maleated lldpe pegma", "maleated LLDPE PEgMA",
            ],

            # Common modifiers and coupling agents
            "O=C1OC(=O)C=C1": ["maleic anhydride", "mah"],
            "C=C(C)C(=O)OC": ["methyl methacrylate", "mma"],
            "CCCCCCCCCCCCCCCCCCN": [
                "octadecylamine", "stearylamine", "stearyl amine", "octadecylammonium",
            ],
            "CCCCCCCCCCCCN": ["dodecylamine", "laurylamine", "lauryl amine"],
            "CCCCCCCCCCCCCCCCCC(=O)O": ["stearic acid"],
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O": ["oleic acid"],
            "CCCCCCCC/C=C\\CCCCCCCC(=O)[O-]": ["oleate anion", "oleate"],
            "CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]": [
                "ctab", "cetyltrimethyl ammonium bromide",
                "hexadecyl trimethyl ammonium bromide", "hexadecyl trimethylammonium bromide",
                "n-hexadecyltrimethylammonium bromine",
            ],
            "CCCCCCCCCCCCCCCCCC[N+](C)(C)CCCCCCCCCCCCCCCCCC.[Cl-]": [
                "dimethyl-dihydrogenatedtallow ammonium chloride",
                "dimethyl dihydrogenated tallow ammonium chloride",
                "dimethyl dehydrogenated tallow ammonium chloride",
                "dimethyl hydrogenated tallow ammonium chloride",
                "dimethyl-dihydrogenated tallow ammonium",
                "dimethyl-dihydrogenated tallow ammonium chloride",
                "dimethyl, dihydrogenated tallow, quaternary ammonium",
                "dimethyl ditallow ammonium",
                "dimethyldistearylammonium cations",
                "dimethyl distearyl ammonium chloride",
                "di-methyl, di-hydrogenated tallow ammonium",
                "dimethyl, two hydrogenated tallow moieties and quaternary ammonium chloride",
                "n-di-methyl dihydro-di-tallow amonia chloride",
                "dimethyl, hydrogenated tallow, 2-ethylhexyl quaternary ammonium",
                "2m2ht", "2m2ht ammonium chloride",
            ],
            "CCCCCCCCCCCCCCCCCC[N+](C)(CCO)CCO.[Cl-]": [
                "methyl-tallow-di(2-hydroxyethyl) ammonium chloride",
                "methyl tallow di(2-hydroxyethyl) ammonium chloride",
                "bis(2-hydroxylethyl)methyl tallowalkyl ammonium cation",
                "bis(2-hydroxyethyl)methyl tallow alkyl ammonium chloride",
                "methyl bis(2-hydroxyethyl) tallow ammonium chloride",
                "mt2etoh", "m2eoh",
            ],
            "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]": [
                "sds", "sodium dodecyl sulfate", "sodium dodecyl sulphate",
                "dodecyl sulfate", "dodecylsulfate",
            ],
            "OS(=O)(=O)O": ["sulfuric acid", "sulphuric acid"],
            "O=C(O)C(=O)O": ["oxalic acid"],
            "B(O)(O)O": ["boric acid"],
            "CCO[Si](CCCN)(OCC)OCC": [
                "aptes", "kh550", "kh-550", "ameo", "3-aminopropyltriethoxysilane",
                "gamma-aminopropyltriethoxysilane", "\u03b3-aminopropyltriethoxysilane",
                "\u00ce\u00b3-aminopropyltriethoxysilane (AMEO)", "kh-550 silane coupling agent",
            ],
            "C=C[Si](OC)(OC)OC": ["vinyl silane", "vinylsilane", "vinyl silane surface treatment"],
            "C=C(C)C(=O)OCCC[Si](OC)(OC)OC": [
                "kh570", "kh-570", "gamma-methylacryloxypropyl trimethoxy silane kh570",
                "3-methacryloxypropyltrimethoxysilane",
                "gamma-methacryloxypropyl trimethoxy silane",
                "gamma-methacryloxypropyltrimethoxysilane",
                "\u03b3-methacryloxypropyl trimethoxy silane",
                "\u00ce\u00b3-methacryloxypropyl trimethoxy silane",
            ],
            "C[Si](C)O": ["pdms", "polydimethylsiloxane", "silicone elastomer", "silicone"],
            "CO[Si](CCCOCC1CO1)(OC)OC": [
                "kh560", "kh-560", "3-glycidoxypropyltrimethoxysilane",
                "gamma-glycidoxypropyltrimethoxysilane",
                "glycidoxypropyltrimethoxysilane", "epoxy silane",
            ],
            "CC1=C(N=C=O)C=CC=C1N=C=O": [
                "tdi", "toluene-2,4-diisocyanate", "toluene diisocyanate",
                "2,4-toluene diisocyanate",
            ],
            "NC1=NC(N)=NC(N)=N1.C=O": [
                "melamine-formaldehyde resin", "melamine formaldehyde resin",
                "melamine-formaldehyde", "melamine formaldehyde", "mf resin",
                "melamine resin",
            ],
            "NC(N)=O.C=O": [
                "urea formaldehyde resin", "urea-formaldehyde resin",
                "urea formaldehyde", "uf resin",
            ],
            "C=CC1=CC=CC=C1": ["ps", "polystyrene", "styrene polymer"],
            "CC1=C(C)C=CC=C1O": [
                "ppo", "poly(phenylene oxide)", "polyphenylene oxide",
                "poly(2,6-dimethyl-1,4-phenylene oxide)",
            ],
            "O": ["water", "h2o"],
            "CN(C)CCOCCO": [
                "dmae", "2-[2-(dimethylamino)ethoxy]ethanol",
                "2-[2-(dimethylamino)-ethoxy] ethanol",
                "2-[2-(dimethylamino)ethoxy]ethanol (dmae)",
                "dmae, 2-[2-(dimethylamino)-ethoxy] ethanol",
            ],
            "CC(C)(OOC(C)(C)c1ccccc1)c1ccccc1": [
                "dcp", "dicumyl peroxide", "di-cumyl peroxide",
            ],
            "C=CCN1C(=O)N(C(=O)N(C1=O)CC=C)CC=C": [
                "taic", "triallyl isocyanurate",
            ],
            "CC(C)(C)C1=CC(=CC(=C1O)C(C)(C)C)CCC(=O)OCC(COC(=O)CCC2=CC(=C(C(=C2)C(C)(C)C)O)C(C)(C)C)(COC(=O)CCC3=CC(=C(C(=C3)C(C)(C)C)O)C(C)(C)C)COC(=O)CCC4=CC(=C(C(=C4)C(C)(C)C)O)C(C)(C)C": [
                "antioxidant 1010", "irganox 1010",
            ],
            "CC(C)(C)C1=CC(=C(C=C1)OP(OC2=C(C=C(C=C2)C(C)(C)C)C(C)(C)C)OC3=C(C=C(C=C3)C(C)(C)C)C(C)(C)C)C(C)(C)C": [
                "antioxidant 168", "irgafos 168",
            ],
            "[O-]C(=O)[O-]": ["carbonate", "carbonate ion"],
            "[Al+3].CC(C)[O-].CC(C)[O-].CC(C)[O-]": [
                "aluminium isopropoxide", "aluminum isopropoxide",
                "aluminium isopropylate", "aluminum isopropylate",
            ],
            "ClP1(Cl)=NP(Cl)(Cl)=NP(Cl)(Cl)=N1": [
                "hccp", "hexachlorocyclotriphosphazene",
                "hexachlorocyclotriphosphazene trimer",
            ],
            "CCCCCCCCCCCCc1ccccc1S(=O)(=O)[O-].[Na+]": [
                "sdbs", "sodium dodecyl benzene sulfonate",
                "sodium dodecylbenzenesulfonate",
            ],
            "CCCCCCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]": [
                "benzyldimethylhexadecylammonium chloride",
                "benzyl dimethyl hexadecyl ammonium chloride",
            ],
            # Bio-based materials
            # Phytic acid (PA/IP6)
            "C1(C(C(C(C(C1OP(=O)(O)O)OP(=O)(O)O)OP(=O)(O)O)OP(=O)(O)O)OP(=O)(O)O)OP(=O)(O)O": ["pa", "ip6", "phytic acid"],
            # Chitosan (CS)
            " OCC1OC(O)C(N)C(O)C1O": ["cs", "chitosan"], 
            " COC1=C(C=CC(=C1)C=CCO)O ": ["lignin", "alkali lignin", "kraft lignin", "kl"],
            # Sodium lignosulfonate (LS)
            "COC1=CC(=CC=C1O)C(S(=O)(=O)[O-])CCO.[Na+]": ["ls", "sodium lignosulfonate", "lignosulfonate"],
            # Tannic acid (TA)
            "Oc1cc(cc(O)c1O)C(=O)Oc2cc(cc(O)c2O)C(=O)OC[C@H]3O[C@@H](OC(=O)c4cc(O)c(O)c(OC(=O)c5cc(O)c(O)c(O)c5)c4)[C@H](OC(=O)c6cc(O)c(O)c(OC(=O)c7cc(O)c(O)c(O)c7)c6)[C@@H](OC(=O)c8cc(O)c(O)c(OC(=O)c9cc(O)c(O)c(O)c9)c8)[C@@H]3OC(=O)c%10cc(O)c(O)c(OC(=O)c%11cc(O)c(O)c(O)c%11)c%10 ": ["ta", "tannic acid"],

            # Sulfonates
            # Potassium diphenyl sulfone sulfonate (KSS)
            " C1=CC=C(C=C1)S(=O)(=O)C2=CC(=CC=C2)S(=O)(=O)[O-].[K+]": ["kss", "potassium diphenyl sulfone sulfonate"],
            # Potassium perfluorobutanesulfonate (PFBS/KPFBS)
            " C(C(C(F)(F)S(=O)(=O)[O-])(F)F)(C(F)(F)F)(F)F.[K+] ": ["pfbs", "kpfbs", "potassium perfluorobutanesulfonate"],

            # MOF/COF materials and ligands
            # ZIF-8
            " CC1=NC=C[N-]1.CC1=NC=C[N-]1.[Zn+2]": ["zif-8", "zeolitic imidazolate framework-8"],
            # UiO-66
            "C1=CC(=CC=C1C(=O)[O-])C(=O)[O-].C1=CC(=CC=C1C(=O)[O-])C(=O)[O-].C1=CC(=CC=C1C(=O)[O-])C(=O)[O-].C1=CC(=CC=C1C(=O)[O-])C(=O)[O-].C1=CC(=CC=C1C(=O)[O-])C(=O)[O-].C1=CC(=CC=C1C(=O)[O-])C(=O)[O-].[OH-].[OH-].[OH-].[OH-].[O-2].[O-2].[Zr+4].[Zr+4].[Zr+4].[Zr+4].[Zr+4].[Zr+4]": ["uio-66", "uio-66 metal-organic framework"],
            # 2-methylimidazole (2-MeIM)
            "CC1=NC=CN1 ": ["2-meim", "2-methylimidazole"],
            # Terephthalic acid (BDC)
            "C1=CC(=CC=C1C(=O)O)C(=O)O": ["bdc", "h2bdc", "pta", "tpa", "terephthalic acid", "1,4-benzenedicarboxylic acid"],
            # 1,3,5-tris(4-aminophenyl)benzene (TAPB)
            "C1=CC(=CC=C1C2=CC(=CC(=C2)C3=CC=C(C=C3)N)C4=CC=C(C=C4)N)N ": ["tapb", "1,3,5-tris(4-aminophenyl)benzene"],
            # 1,3,5-triformylbenzene (TFB)
            "O=Cc1cc(C=O)cc(C=O)c1": ["tfb", "bta", "1,3,5-triformylbenzene", "benzene-1,3,5-tricarboxaldehyde"]
}


knowledge_base = {
    str(smiles).strip(): [str(name).strip() for name in names]
    for smiles, names in knowledge_base.items()
}


SPECIAL_MATERIAL_RULES = (
    {
        "canonical_name": "EG",
        "aliases": ("eg", "eg1", "eg2", "expandable graphite", "expanded graphite"),
        "material_class": "Expandable Graphite",
        "proxy_smiles": "[C].[O-]S(=O)(=O)[O-]",
        "structure_proxy": "expandable_graphite_intercalated_material",
    },
    {
        "canonical_name": "GO",
        "aliases": ("go", "graphene oxide"),
        "material_class": "Graphene Oxide",
        "structure_proxy": "graphene_oxide_sheet",
    },
    {
        "canonical_name": "Graphite Oxide",
        "aliases": ("graphite oxide",),
        "material_class": "Graphite Oxide",
        "structure_proxy": "graphite_oxide_layer",
    },
    {
        "canonical_name": "Graphene",
        "aliases": ("graphene",),
        "material_class": "Graphenic Carbon Material",
        "structure_proxy": "graphene_sheet",
    },
    {
        "canonical_name": "Graphite",
        "aliases": ("graphite",),
        "material_class": "Graphitic Carbon Material",
        "structure_proxy": "graphite_layered_solid",
    },
    {
        "canonical_name": "Natural Flake Graphite",
        "aliases": ("natural flake graphite",),
        "material_class": "Graphitic Carbon Material",
        "structure_proxy": "natural_flake_graphite",
    },
    {
        "canonical_name": "CNT",
        "aliases": ("cnt", "cnts", "carbon nanotube", "carbon nanotubes"),
        "material_class": "Carbon Nanotube Material",
        "structure_proxy": "carbon_nanotube",
    },
    {
        "canonical_name": "MWCNT",
        "aliases": (
            "mwnt", "mwnts", "mwcnt", "mwcnts", "multi-walled carbon nanotube",
            "multi-walled carbon nanotubes", "multi walled carbon nanotube",
            "multi walled carbon nanotubes",
        ),
        "material_class": "Multi-Walled Carbon Nanotube Material",
        "structure_proxy": "multi_walled_carbon_nanotube",
    },
    {
        "canonical_name": "SWCNT",
        "aliases": (
            "swnt", "swnts", "swcnt", "swcnts", "single-walled carbon nanotube",
            "single-walled carbon nanotubes", "single walled carbon nanotube",
            "single walled carbon nanotubes",
        ),
        "material_class": "Single-Walled Carbon Nanotube Material",
        "structure_proxy": "single_walled_carbon_nanotube",
    },
    {
        "canonical_name": "Carbon Black",
        "aliases": ("carbon black",),
        "material_class": "Carbon Black",
        "structure_proxy": "carbon_black_particulate",
    },
    {
        "canonical_name": "MMT",
        "aliases": ("mmt", "montmorillonite", "montmorillonite clay"),
        "material_class": "Montmorillonite Clay Mineral",
        "structure_proxy": "montmorillonite_layered_clay",
    },
    {
        "canonical_name": "Na-MMT",
        "aliases": ("na-mmt", "na mmt", "sodium montmorillonite"),
        "material_class": "Sodium Montmorillonite Clay",
        "structure_proxy": "sodium_montmorillonite",
    },
    {
        "canonical_name": "OMMT",
        "aliases": (
            "ommt", "omt", "organic mmt", "organic montmorillonite",
            "organically modified montmorillonite",
        ),
        "material_class": "Organically Modified Montmorillonite",
        "structure_proxy": "organically_modified_montmorillonite",
    },
    {
        "canonical_name": "Organoclay",
        "aliases": ("organoclay", "organic clay"),
        "material_class": "Organically Modified Clay",
        "structure_proxy": "organoclay",
    },
    {
        "canonical_name": "Bentonite",
        "aliases": ("bentonite",),
        "material_class": "Bentonite Clay Mineral",
        "structure_proxy": "bentonite_clay",
    },
    {
        "canonical_name": "Cloisite 20A",
        "aliases": ("cloisite 20a", "cloisite-20a"),
        "material_class": "Commercial Organoclay",
        "structure_proxy": "cloisite_20a",
    },
    {
        "canonical_name": "Cloisite 30B",
        "aliases": ("cloisite 30b", "cloisite-30b"),
        "material_class": "Commercial Organoclay",
        "structure_proxy": "cloisite_30b",
    },
    {
        "canonical_name": "Fe-MMT",
        "aliases": ("fe-montmorillonite", "fe montmorillonite", "fe-mmt", "fe mmt"),
        "material_class": "Iron-Modified Montmorillonite",
        "structure_proxy": "iron_modified_montmorillonite",
    },
    {
        "canonical_name": "LDH",
        "aliases": ("ldh", "ldhs", "layered double hydroxide", "layered double hydroxides"),
        "material_class": "Layered Double Hydroxide",
        "structure_proxy": "generic_layered_double_hydroxide",
    },
    {
        "canonical_name": "MgAl-LDH",
        "aliases": (
            "mgal layered double hydroxide", "mg-al layered double hydroxide",
            "mgal-ldh", "mg-al ldh", "mg al ldh", "mg/al ldh", "mg-al ldhs",
            "mgal ldhs", "mg-al layered double hydroxides", "mgal layered double hydroxides",
            "magnesium aluminum layered double hydroxide",
            "magnesium aluminum layered double hydroxides", "hydrotalcite",
        ),
        "material_class": "Magnesium-Aluminum Layered Double Hydroxide",
        "structure_proxy": "mg_al_layered_double_hydroxide",
    },
    {
        "canonical_name": "ZnAl-LDH",
        "aliases": (
            "zn-al layered double hydroxide", "zn-al layered double hydroxides",
            "zn-al ldh", "zn-al ldhs", "zn al ldh", "zn/al ldh",
            "zinc aluminum layered double hydroxide",
            "zinc aluminum layered double hydroxides",
        ),
        "material_class": "Zinc-Aluminum Layered Double Hydroxide",
        "structure_proxy": "zn_al_layered_double_hydroxide",
    },
    {
        "canonical_name": "NiAl-LDH",
        "aliases": (
            "ni-al layered double hydroxide", "ni-al layered double hydroxides",
            "ni-al ldh", "ni-al ldhs", "ni al ldh", "ni/al ldh",
            "nickel aluminum layered double hydroxide",
            "nickel aluminum layered double hydroxides",
        ),
        "material_class": "Nickel-Aluminum Layered Double Hydroxide",
        "structure_proxy": "ni_al_layered_double_hydroxide",
    },
    {
        "canonical_name": "Isostearic Acid",
        "aliases": ("isostearic acid",),
        "material_class": "Branched C18 Fatty Acid Isomer Mixture",
        "structure_proxy": "isostearic_acid_isomer_mixture",
    },
)


HIGH_AUTHORITY_SOURCE_RE = re.compile(
    r"(manual|human|curated|verified|validated|reviewed|mllm|sniped|success)",
    re.IGNORECASE,
)
LOW_AUTHORITY_SOURCE_RE = re.compile(
    r"^(?:\s*$|#|no dictionary match|pending visual extraction|expert dictionary|"
    r"material class|api error|failed)",
    re.IGNORECASE,
)


def normalize_alias(value):
    text = str(value or "").strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(text.split())


SPECIAL_ALIAS_INDEX = {
    normalize_alias(alias): rule
    for rule in SPECIAL_MATERIAL_RULES
    for alias in rule["aliases"]
}
SPECIAL_ALIASES_BY_LENGTH = sorted(SPECIAL_ALIAS_INDEX, key=len, reverse=True)

def find_special_material_rule(raw_name, allow_variant=False):
    if pd.isna(raw_name) or str(raw_name).strip() in {"", "#"}:
        return None

    raw_key = normalize_alias(raw_name)
    exact = SPECIAL_ALIAS_INDEX.get(raw_key)
    if exact is not None or not allow_variant:
        return exact

    for alias in SPECIAL_ALIASES_BY_LENGTH:
        match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", raw_key)
        if not match:
            continue
        remainder = raw_key[:match.start()] + raw_key[match.end():]
        remainder = re.sub(r"\b(?:wt|vol|phr|part|parts|percent)\b|%", "", remainder)
        if not re.search(r"[a-z]", remainder):
            return SPECIAL_ALIAS_INDEX[alias]
    return None


def special_rule_details(rule, match_label):
    proxy_smiles = rule.get("proxy_smiles", "#")
    if proxy_smiles == "#":
        source = f"Material Class (No Representative SMILES; {match_label})"
    else:
        source = f"Curated Representative SMILES Proxy ({match_label})"
    return {
        "canonical_name": rule["canonical_name"],
        "smiles": proxy_smiles,
        "source": source,
        "material_class": rule["material_class"],
        "structure_proxy": rule["structure_proxy"],
        "structure_proxy_source": "Conservative Material Alias Mapping",
        "is_molecular": False,
    }


def knowledge_base_exact_match(raw_name, mapping):
    raw_key = normalize_alias(raw_name)
    for smiles, names in mapping.items():
        for dict_name in names:
            if raw_key == normalize_alias(dict_name):
                return names[0].upper(), str(smiles).strip()
    return None, None


def knowledge_base_strict_match(raw_name, mapping):
    raw_key = normalize_alias(raw_name)
    for smiles, names in mapping.items():
        for dict_name in names:
            alias = normalize_alias(dict_name)
            if alias not in raw_key:
                continue
            remainder = raw_key.replace(alias, "", 1)
            if not re.search(r"[a-z]", remainder):
                return names[0].upper(), str(smiles).strip(), True
    return raw_name, "#", False


def get_exact_match(raw_name, mapping):
    special_rule = find_special_material_rule(raw_name, allow_variant=False)
    if special_rule is not None:
        details = special_rule_details(special_rule, "Exact Alias")
        return details["canonical_name"], details["smiles"]
    return knowledge_base_exact_match(raw_name, mapping)


def map_expert_dict_strict(raw_name, mapping):
    special_rule = find_special_material_rule(raw_name, allow_variant=True)
    if special_rule is not None:
        details = special_rule_details(special_rule, "Strict Variant")
        return details["canonical_name"], details["smiles"], True
    return knowledge_base_strict_match(raw_name, mapping)


def match_name_details(raw_name):
    special_rule = find_special_material_rule(raw_name, allow_variant=False)
    if special_rule is not None:
        return special_rule_details(special_rule, "Exact Alias")

    standard_name, smiles = knowledge_base_exact_match(raw_name, knowledge_base)
    if standard_name is not None:
        return {
            "canonical_name": standard_name,
            "smiles": smiles,
            "source": "Expert Dictionary (Exact Match)",
            "material_class": "Molecular or Discrete Ionic Material",
            "structure_proxy": "#",
            "structure_proxy_source": "Not Applicable (Molecular SMILES Available)",
            "is_molecular": True,
        }

    special_rule = find_special_material_rule(raw_name, allow_variant=True)
    if special_rule is not None:
        return special_rule_details(special_rule, "Strict Variant")

    standard_name, smiles, matched = knowledge_base_strict_match(raw_name, knowledge_base)
    if matched:
        return {
            "canonical_name": standard_name,
            "smiles": smiles,
            "source": "Expert Dictionary (Strict Variants)",
            "material_class": "Molecular or Discrete Ionic Material",
            "structure_proxy": "#",
            "structure_proxy_source": "Not Applicable (Molecular SMILES Available)",
            "is_molecular": True,
        }
    return None


TARGET_FIELD_SPECS = [
    # Final wide table fields
    ("flame_retardant_1", "flame_retardant_1_figure_locator", "flame_retardant_1_smiles", "flame_retardant_1_smiles_source"),
    ("flame_retardant_2", "flame_retardant_2_figure_locator", "flame_retardant_2_smiles", "flame_retardant_2_smiles_source"),
    ("flame_retardant_1_modifier", "flame_retardant_1_modifier_figure_locator", "flame_retardant_1_modifier_smiles", "flame_retardant_1_modifier_smiles_source"),
    ("flame_retardant_2_modifier", "flame_retardant_2_modifier_figure_locator", "flame_retardant_2_modifier_smiles", "flame_retardant_2_modifier_smiles_source"),
    ("flame_retardant_1_base_or_core_components", None, "flame_retardant_1_base_or_core_components_smiles", "flame_retardant_1_base_or_core_components_smiles_source"),
    ("flame_retardant_2_base_or_core_components", None, "flame_retardant_2_base_or_core_components_smiles", "flame_retardant_2_base_or_core_components_smiles_source"),
    ("flame_retardant_1_shell_grafted_or_modified_components", None, "flame_retardant_1_shell_grafted_or_modified_components_smiles", "flame_retardant_1_shell_grafted_or_modified_components_smiles_source"),
    ("flame_retardant_2_shell_grafted_or_modified_components", None, "flame_retardant_2_shell_grafted_or_modified_components_smiles", "flame_retardant_2_shell_grafted_or_modified_components_smiles_source"),
    ("flame_retardant_1_modified_product_name", "flame_retardant_1_figure_locator", "flame_retardant_1_modified_product_smiles", "flame_retardant_1_modified_product_smiles_source"),
    ("flame_retardant_2_modified_product_name", "flame_retardant_2_figure_locator", "flame_retardant_2_modified_product_smiles", "flame_retardant_2_modified_product_smiles_source"),
    ("other_flame_retardant_or_synergist_components", None, "other_flame_retardant_or_synergist_components_smiles", "other_flame_retardant_or_synergist_components_smiles_source"),
    # Long-table fields from eva_split_pipeline.py
    ("entity_name", "entity_figure_locator", "entity_smiles", "entity_smiles_source"),
    ("part_name", "part_figure_locator", "part_smiles", "part_smiles_source"),
    ("component_name", "component_figure_locator", "component_smiles", "component_smiles_source"),
    ("composite_or_modified_product_name", "component_figure_locator", "composite_or_modified_product_smiles", "composite_or_modified_product_smiles_source"),
    ("modified_component", None, "modified_component_smiles", "modified_component_smiles_source"),
]


def split_name_tokens(raw_name):
    if pd.isna(raw_name) or str(raw_name).strip() in {"", "#", "nan"}:
        return []
    text = str(raw_name).strip()
    # Do not split chemical locants or chemical names containing commas.
    # English comma is treated as a list separator only for obvious "A, B" lists.
    return [
        item.strip()
        for item in re.split(r"\s*(?:;|\||/|\+)\s*|,\s+(?=[A-Z#])", text)
        if item.strip() and item.strip() != "#"
    ]


def dict_match_name(raw_name):
    details = match_name_details(raw_name)
    if details is not None:
        return details["canonical_name"], details["smiles"], details["source"]
    return raw_name, "#", None


def dict_match_multi_details(raw_name):
    direct_match = match_name_details(raw_name)
    if direct_match is not None:
        return direct_match

    tokens = split_name_tokens(raw_name)
    if not tokens:
        return None

    matched_names = []
    matched_smiles = []
    matched_sources = []
    material_classes = []
    structure_proxies = []
    structure_proxy_sources = []
    any_matched = False
    for token in tokens:
        details = match_name_details(token)
        matched_names.append(details["canonical_name"] if details else token)
        matched_smiles.append(details["smiles"] if details else "#")
        matched_sources.append(details["source"] if details else "No Dictionary Match")
        material_classes.append(details["material_class"] if details else "#")
        structure_proxies.append(details["structure_proxy"] if details else "#")
        structure_proxy_sources.append(details["structure_proxy_source"] if details else "#")
        any_matched = any_matched or details is not None

    if not any_matched:
        return None
    return {
        "canonical_name": "; ".join(matched_names),
        "smiles": "; ".join(matched_smiles),
        "source": "; ".join(matched_sources),
        "material_class": "; ".join(material_classes),
        "structure_proxy": "; ".join(structure_proxies),
        "structure_proxy_source": "; ".join(structure_proxy_sources),
        "is_molecular": all(
            details is not None and details["is_molecular"]
            for details in (match_name_details(token) for token in tokens)
        ),
    }


def dict_match_multi(raw_name):
    details = dict_match_multi_details(raw_name)
    if details is None:
        return raw_name, "#", None
    return details["canonical_name"], details["smiles"], details["source"]


def is_empty_cell(value):
    return pd.isna(value) or str(value).strip().lower() in {"", "#", "nan", "none", "null", "n/a"}


def ensure_column(df, column_name):
    if column_name not in df.columns:
        df[column_name] = pd.NA
    else:
        df[column_name] = df[column_name].astype("object")


def target_metadata_columns(name_col):
    return {
        "original_name": f"{name_col}_original",
        "material_class": f"{name_col}_material_category",
        "structure_proxy": f"{name_col}_structure_proxy",
        "structure_proxy_source": f"{name_col}_structure_proxy_source",
    }


def source_is_high_authority(value):
    return bool(HIGH_AUTHORITY_SOURCE_RE.search(normalize_for_rule(value)))


def should_preserve_existing(smiles_value, source_value):
    if source_is_high_authority(source_value):
        return True
    if is_empty_cell(smiles_value):
        return False
    source_text = normalize_for_rule(source_value)
    if not source_text:
        return True
    return not bool(LOW_AUTHORITY_SOURCE_RE.search(source_text))


def available_target_specs(df):
    return [spec for spec in TARGET_FIELD_SPECS if spec[0] in df.columns]


VISUAL_ROLE_HINTS = {
    "main_flame_retardant",
    "synergistic_flame_retardant",
    "functional_filler",
    "modifier",
    "shell",
    "coating_layer",
    "grafted_moiety",
    "surface_ligand",
    "coupling_agent",
    "complexing_agent",
    "synergistic_part",
}

VISUAL_POSITIVE_RE = re.compile(
    r"(graft|grafted|grafting|synthesi[sz]ed|synthesis|synthetic|reaction|scheme|"
    r"new\s+(?:flame\s+)?retardant|novel|designed|derived|derivative|intermediate|"
    r"product\s+structure|chemical\s+structure|molecular\s+structure|repeat(?:ing)?\s+unit|"
    r"dopo|phosphazene|triazine|triazole|hyperbranched|oligomer|polymeric\s+flame|"
    r"organophosph|p-n|p\s*/\s*n|p\s*-?\s*n|p\s*-?\s*n\s*-?\s*s)",
    re.IGNORECASE,
)

HARD_VISUAL_SKIP_RE = re.compile(
    r"(core[-\s]?shell|shell[-\s]?core|microencapsulat|encapsulat|coated|coating|"
    r"surface[-\s]?treated|surface\s+modified|intercalat|doped|in[-\s]?situ\s+grown|"
    r"morphology|sem|tem|xrd|tga|ftir|dtg|xps|dsc|eds|mapping)",
    re.IGNORECASE,
)

SOFT_VISUAL_SKIP_RE = re.compile(
    r"(hybrid|composite|complex|particle|nanoparticle|filler)",
    re.IGNORECASE,
)

STRUCTURE_LOCATOR_RE = re.compile(
    r"(structure|chemical|molecular|reaction|scheme|synthesis|synthetic|formula)",
    re.IGNORECASE,
)

NON_STRUCTURE_LOCATOR_RE = re.compile(
    r"(sem|tem|morphology|xrd|tga|dtg|ftir|spectrum|spectra|xps|dsc|eds|mapping)",
    re.IGNORECASE,
)

COMMON_NON_MOLECULAR_RE = re.compile(
    r"\b(?:ath|mdh|mh|app|mpp|mca|eg|cnts?|mwcnts?|swcnts?|go|rgo|graphene|"
    r"graphite|zinc borate|zb|ldh|mmt|montmorillonite|clay|talc|sio2|silica|"
    r"tio2|al2o3|mgo|zno|fe2o3|red phosphorus|rp|mxene|zif|uio|mof|cof|"
    r"magnesium hydroxide|aluminum hydroxide|aluminium hydroxide|ammonium polyphosphate)\b",
    re.IGNORECASE,
)

ORGANIC_STRUCTURE_MARKER_RE = re.compile(
    r"(dopo|phosphazene|triazine|triazole|silane|kh-?\d+|ligand|monomer|"
    r"phosphate|phosphonate|phosphinate|piperazine|melamine|chitosan|lignin|"
    r"tannic|hyperbranched|oligomer|polymer|organic)",
    re.IGNORECASE,
)


def normalize_for_rule(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def joined_row_context(row):
    context_columns = [
        "entity_name", "entity_role", "entity_type", "entity_material_type", "entity_profile", "entity_figure_locator",
        "part_name", "part_role", "part_relation", "modification_method", "part_figure_locator", "part_notes",
        "component_name", "component_figure_locator", "composite_or_modified_product_name",
    ]
    return " | ".join(normalize_for_rule(row.get(col)) for col in context_columns if col in row.index)


def has_image_id(locator):
    return bool(re.search(r"\bID\s*:", normalize_for_rule(locator), re.IGNORECASE))


def role_is_visual_relevant(row):
    role_text = " ".join(
        normalize_for_rule(row.get(col)).lower()
        for col in ["entity_role", "part_role", "part_relation"]
        if col in row.index
    )
    if not role_text:
        return True
    return any(role in role_text for role in VISUAL_ROLE_HINTS)


def should_mark_pending_visual(row, name_col, locator_col):
    """Strict gate: only new/grafted/synthesized organic FR-like structures go to costly vision."""
    target_name = normalize_for_rule(row.get(name_col))
    if not target_name or target_name in {"#", "nan", "None"}:
        return False, "empty target"

    locator = normalize_for_rule(row.get(locator_col)) if locator_col and locator_col in row.index else ""
    if not has_image_id(locator):
        return False, "no image locator"

    context = f"{target_name} | {locator} | {joined_row_context(row)}"
    positive = bool(VISUAL_POSITIVE_RE.search(context))
    structure_locator = bool(STRUCTURE_LOCATOR_RE.search(locator))
    non_structure_locator = bool(NON_STRUCTURE_LOCATOR_RE.search(locator))
    organic_marker = bool(ORGANIC_STRUCTURE_MARKER_RE.search(context))
    common_non_molecular = bool(COMMON_NON_MOLECULAR_RE.search(target_name))
    hard_skip = bool(HARD_VISUAL_SKIP_RE.search(context))
    soft_skip = bool(SOFT_VISUAL_SKIP_RE.search(context))

    if not role_is_visual_relevant(row):
        return False, "role not visual target"

    # APP@SiO2, coated MDH, ZIF/LDH, core-shell particles etc. are represented by parts
    # and modification method; they do not have one meaningful molecular SMILES.
    if hard_skip and not re.search(r"(graft|grafted)", context, re.IGNORECASE):
        return False, "composite/surface/morphology evidence, not a single molecule"

    if soft_skip and not (positive and organic_marker):
        return False, "composite/particle context without clear organic structure"

    if common_non_molecular and not organic_marker:
        return False, "common inorganic/mineral filler"

    if non_structure_locator and not structure_locator:
        return False, "image is not chemical-structure evidence"

    if not (positive or structure_locator):
        return False, "no synthesis/structure signal"

    if not organic_marker and not re.search(r"(graft|grafted|synthesi[sz]ed)", context, re.IGNORECASE):
        return False, "not enough organic/new-product signal"

    return True, "new/grafted/synthesized structure candidate"


def clean_dataframe(dataframe):
    cleaned = dataframe.copy(deep=True)
    target_specs = available_target_specs(cleaned)

    for name_col, _, smiles_col, source_col in target_specs:
        ensure_column(cleaned, smiles_col)
        ensure_column(cleaned, source_col)
        for metadata_col in target_metadata_columns(name_col).values():
            ensure_column(cleaned, metadata_col)

    for index, row in cleaned.iterrows():
        for name_col, locator_col, smiles_col, source_col in target_specs:
            current_name = row.get(name_col)
            if is_empty_cell(current_name):
                continue

            metadata_cols = target_metadata_columns(name_col)
            stored_original = row.get(metadata_cols["original_name"])
            mapping_name = current_name if is_empty_cell(stored_original) else stored_original
            if is_empty_cell(stored_original):
                cleaned.at[index, metadata_cols["original_name"]] = str(current_name).strip()

            details = dict_match_multi_details(mapping_name)
            existing_smiles = cleaned.at[index, smiles_col]
            existing_source = cleaned.at[index, source_col]
            preserve_existing = should_preserve_existing(existing_smiles, existing_source)

            if details is not None:
                cleaned.at[index, metadata_cols["material_class"]] = details["material_class"]
                cleaned.at[index, metadata_cols["structure_proxy"]] = details["structure_proxy"]
                cleaned.at[index, metadata_cols["structure_proxy_source"]] = details["structure_proxy_source"]
                if not preserve_existing:
                    cleaned.at[index, name_col] = details["canonical_name"]
                    cleaned.at[index, smiles_col] = str(details["smiles"]).strip()
                    cleaned.at[index, source_col] = details["source"]
                continue

            if preserve_existing:
                continue

            img_locator = str(row.get(locator_col, "")) if locator_col and locator_col in cleaned.columns else ""
            image_id_present = has_image_id(img_locator)
            pending, reason = should_mark_pending_visual(row, name_col, locator_col)
            if pending:
                cleaned.at[index, source_col] = "Pending Visual Extraction"
            elif image_id_present:
                if is_empty_cell(existing_source):
                    cleaned.at[index, source_col] = f"No Dictionary Match; Visual Skipped: {reason}"
            elif is_empty_cell(existing_source):
                cleaned.at[index, source_col] = "No Dictionary Match"

    return cleaned


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(path, encoding="utf-8-sig", dtype=object, keep_default_na=False)
    elif suffix == ".tsv":
        dataframe = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=object, keep_default_na=False)
    elif suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path, dtype=object, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")
    return canonicalize_dataframe_columns(dataframe, context=str(path))


def write_table(dataframe, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if suffix == ".tsv":
        dataframe.to_csv(path, index=False, sep="\t", encoding="utf-8-sig")
        return
    if suffix == ".xlsx":
        dataframe.to_excel(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path.suffix}. Use .csv, .tsv, or .xlsx.")


def process_single_dataset(input_path, output_path=None, dry_run=False):
    print(f"Loading extracted data: {input_path}")
    dataframe = read_table(input_path)
    target_specs = available_target_specs(dataframe)
    print(f"Starting hierarchical dictionary matching across {len(target_specs)} chemical field types...")
    print("Fields: " + ", ".join(spec[0] for spec in target_specs))
    cleaned = clean_dataframe(dataframe)

    if dry_run:
        print(f"dry_run=true; rows={len(cleaned)}; no output written")
        return cleaned
    if output_path is None:
        raise ValueError("output_path is required unless dry_run=True")
    write_table(cleaned, output_path)
    print(f"Dictionary normalization complete; saved to: {output_path}")
    return cleaned


def run_self_tests():
    mapping_cases = {
        "expandable graphite": ("EG", "[C].[O-]S(=O)(=O)[O-]"),
        "EG2": ("EG", "[C].[O-]S(=O)(=O)[O-]"),
        "MWNT": ("MWCNT", "#"),
        "carbon black": ("Carbon Black", "#"),
        "OMMT": ("OMMT", "#"),
        "Zn-Al LDH": ("ZnAl-LDH", "#"),
    }
    observed = {}
    for raw_name, (expected_name, expected_smiles) in mapping_cases.items():
        details = match_name_details(raw_name)
        assert details is not None, f"No mapping for {raw_name}"
        assert details["canonical_name"] == expected_name, (raw_name, details)
        assert details["smiles"] == expected_smiles, (raw_name, details)
        assert not details["is_molecular"], (raw_name, details)
        observed[raw_name] = details["canonical_name"]

    assert observed["MWNT"] != "GO"
    assert observed["carbon black"] != "GO"
    assert observed["OMMT"] != "MMT"
    assert observed["Zn-Al LDH"] != "MgAl-LDH"

    stearic = match_name_details("stearic acid")
    oleic = match_name_details("oleic acid")
    assert stearic is not None and oleic is not None
    assert stearic["smiles"] != oleic["smiles"]
    assert "C=C" in oleic["smiles"] and "C=C" not in stearic["smiles"]

    protected = pd.DataFrame(
        [{
            "entity_name": "MWNT",
            "entity_smiles": "manual-structure-value",
            "entity_smiles_source": "Manual Verified",
        }]
    )
    protected_result = clean_dataframe(protected)
    assert protected_result.at[0, "entity_name"] == "MWNT"
    assert protected_result.at[0, "entity_smiles"] == "manual-structure-value"
    assert protected_result.at[0, "entity_smiles_source"] == "Manual Verified"
    assert protected_result.at[0, "entity_name_original"] == "MWNT"

    conservative = clean_dataframe(pd.DataFrame([{"entity_name": "carbon black"}]))
    assert conservative.at[0, "entity_name"] == "Carbon Black"
    assert conservative.at[0, "entity_smiles"] == "#"
    assert "No Representative SMILES" in conservative.at[0, "entity_smiles_source"]
    assert conservative.at[0, "entity_name_material_category"] == "Carbon Black"

    print("self_test=passed")
    for raw_name, canonical_name in observed.items():
        print(f"  {raw_name} -> {canonical_name} | SMILES={mapping_cases[raw_name][1]}")
    print(f"  stearic acid -> {stearic['smiles']}")
    print(f"  oleic acid -> {oleic['smiles']}")
    print("  high-authority Manual/MLLM/verified values are protected")
    return True


def process_eva_output_dir(input_dir, output_dir=None, in_place=False, include_wide=False, dry_run=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    tables = [
        "formulation_entities.xlsx",
        "formulation_entity_parts.xlsx",
    ]
    if include_wide:
        tables.append("flame_retardant_data_optimized_final.xlsx")

    processed = []
    for filename in tables:
        input_path = input_dir / filename
        if not input_path.exists():
            print(f"Not found; skipping: {input_path}")
            continue

        if in_place:
            output_path = input_path
        else:
            output_path = output_dir / f"{input_path.stem}_dict.xlsx"

        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        process_single_dataset(str(input_path), str(output_path), dry_run=dry_run)
        processed.append(output_path)

    if not processed:
        raise FileNotFoundError(f"No processable EVA table found: {input_dir}")

    print("\nEVA dictionary stage complete:")
    for path in processed:
        print(f"   - {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean flame-retardant names/SMILES with the expert dictionary.")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "outputs" / "pipeline" / "formulation_entities.xlsx"),
        help="Input .csv/.tsv/.xlsx/.xls table produced by extraction or raw replay.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "pipeline_normalized" / "formulation_entities.xlsx"),
        help="Output .csv/.tsv/.xlsx after dictionary cleaning.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="EVA pipeline output directory. Processes formulation_entities.xlsx and formulation_entity_parts.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for --input-dir mode. Default: same as input-dir.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite EVA entity/part tables in --input-dir mode so eva_split_pipeline.py --task merge can use them directly.",
    )
    parser.add_argument(
        "--include-wide",
        action="store_true",
        help="Also dictionary-clean the final wide table. Normally not needed; wide table should be regenerated from enriched entity/part tables.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Clean in memory and print a summary without writing output files.")
    parser.add_argument("--self-test", action="store_true", help="Run conservative mapping assertions without reading or writing data files.")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
    elif args.input_dir:
        process_eva_output_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            in_place=args.in_place,
            include_wide=args.include_wide,
            dry_run=args.dry_run,
        )
    else:
        process_single_dataset(
            input_path=args.input,
            output_path=args.output,
            dry_run=args.dry_run,
        )
