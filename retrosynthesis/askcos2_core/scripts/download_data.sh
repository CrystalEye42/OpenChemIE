#!/bin/bash

mkdir -p ./data/db

mkdir -p ./data/db/buyables
if [ ! -f ./data/db/buyables/buyables.json.gz ]; then
    echo "./data/db/buyables/buyables.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/buyables/buyables.json.gz \
      "https://www.dropbox.com/scl/fi/jaqo5r8jji7n19ijrohj1/buyables_ori_prop_new2.json.gz?rlkey=lnbcj7t1ygrjgqi3g7a7vshz2&st=krp61fnf&dl=0"
    echo "buyables.json.gz Downloaded."
fi

if [ ! -f ./data/db/buyables/chembridge_buyables.json.gz ]; then
    echo "./data/db/buyables/chembridge_buyables.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/buyables/chembridge_buyables.json.gz \
      "https://www.dropbox.com/scl/fi/v5jayotulkdueiebck39g/chembridge_buyables_prop_new.json.gz?rlkey=zj3ncebc29ohzmhenj8fmeeo2&dl=1"
    echo "chembridge_buyables.json.gz Downloaded."
fi

if [ ! -f ./data/db/buyables/mcule_buyables_fd2.json.gz ]; then
    echo "./data/db/buyables/mcule_buyables_fd2.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/buyables/mcule_buyables_fd2.json.gz \
      "https://www.dropbox.com/scl/fi/1ikitf6n1lylyq3eroqxp/mcule_buyables_fd2_prop_new.json.gz?rlkey=2rjslc4n7gvvplsznx05vom4i&dl=1"
    echo "mcule_buyables_fd2.json.gz Downloaded."
fi

mkdir -p ./data/db/historian
if [ ! -f ./data/db/historian/chemicals.json.gz ]; then
    echo "./data/db/historian/chemicals.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/historian/chemicals.json.gz \
      "https://www.dropbox.com/scl/fi/q1895i8f3rnjywa1n6d35/chemicals.json.gz?rlkey=vm63mwm29h1oc63pzqmk9qm87&dl=1"
    echo "chemicals.json.gz Downloaded."
fi

if [ ! -f ./data/db/historian/historian.bkms_metabolic.json.gz ]; then
    echo "./data/db/historian/historian.bkms_metabolic.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/historian/historian.bkms_metabolic.json.gz \
      "https://www.dropbox.com/scl/fi/rdp6h7zbt4r10g2ks7qjc/historian.bkms_metabolic.json.gz?rlkey=yciksi7hosoqdon23o0zrkty6&dl=1"
    echo "historian.bkms_metabolic.json.gz Downloaded."
fi

if [ ! -f ./data/db/historian/historian.pistachio.json.gz ]; then
    echo "./data/db/historian/historian.pistachio.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/historian/historian.pistachio.json.gz \
      "https://www.dropbox.com/scl/fi/uf4kn1ltuyjrt78nrgkej/historian.pistachio.json.gz?rlkey=h6tmamfp5hdszon7npnqbn6bm&dl=1"
    echo "historian.pistachio.json.gz Downloaded."
fi

if [ ! -f ./data/db/historian/reactions.bkms_metabolic.json.gz ]; then
    echo "./data/db/historian/reactions.bkms_metabolic.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/historian/reactions.bkms_metabolic.json.gz \
      "https://www.dropbox.com/scl/fi/67tc0zicbe5hpay9cisi1/reactions.bkms_metabolic.json.gz?rlkey=kx5577t58enil05nqembqpa96&dl=1"
    echo "reactions.bkms_metabolic.json.gz Downloaded."
fi

if [ ! -f ./data/db/historian/reactions.cas.min.json.gz ]; then
    if [ -n "${DROPBOX_LINK_PASSWORD}" ]; then
        echo "./data/db/historian/reactions.cas.min.json.gz not found. Downloading.."
        curl -X POST https://content.dropboxapi.com/2/sharing/get_shared_link_file \
          --header "Authorization: Bearer ${DROPBOX_ACCESS_TOKEN}" \
          --header "Dropbox-API-Arg: {\"path\":\"/reactions.cas.min.json.gz\",\"url\":\"https://www.dropbox.com/scl/fi/qa0uu6co3g0kp1988sgxe/reactions.cas.min.json.gz?rlkey=dj3cu1y6uia10pm7tt9pnpza7&dl=0\", \"link_password\":\"${DROPBOX_LINK_PASSWORD}\"}" \
          -o ./data/db/historian/reactions.cas.min.json.gz
        echo "reactions.cas.min.json.gz Downloaded."
    fi
fi

if [ ! -f ./data/db/historian/reactions.pistachio.json.gz ]; then
    echo "./data/db/historian/reactions.pistachio.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/historian/reactions.pistachio.json.gz \
      "https://www.dropbox.com/scl/fi/1tzblq7b92ld8q5cy7w19/reactions.pistachio.min.json.gz?rlkey=p2ug9qvj78q1efdx3xmfabv3h&dl=1"
    echo "reactions.pistachio.json.gz Downloaded."
fi

if [ ! -f ./data/db/historian/reactions.USPTO_FULL.json.gz ]; then
    echo "./data/db/historian/reactions.USPTO_FULL.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/historian/reactions.USPTO_FULL.json.gz \
      "https://www.dropbox.com/scl/fi/19h8ip139l2o1dyu95uog/reactions.USPTO_FULL.json.gz?rlkey=6lyepgx5oiei0gct6i5miuyuy&dl=1"
    echo "reactions.USPTO_FULL.json.gz Downloaded."
fi

mkdir -p ./data/db/references
if [ ! -f ./data/db/references/site_selectivity.refs.json.gz ]; then
    echo "./data/db/references/site_selectivity.refs.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/references/site_selectivity.refs.json.gz \
      "https://www.dropbox.com/scl/fi/y9qxa5js03gf6unaudu05/site_selectivity.refs.json.gz?rlkey=sg8uxjo92rsxkek6t2d8vwp69&dl=1"
    echo "site_selectivity.refs.json.gz Downloaded."
fi

mkdir -p ./data/db/templates
if [ ! -f ./data/db/templates/forward.templates.json.gz ]; then
    echo "./data/db/templates/forward.templates.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/templates/forward.templates.json.gz \
      "https://www.dropbox.com/scl/fi/7u0dbhknyjcxd8khgqbh4/forward.templates.json.gz?rlkey=whcy472tmzr6k02n2jnm0b9ei&dl=1"
    echo "forward.templates.json.gz Downloaded."
fi

if [ ! -f ./data/db/templates/retro.templates.bkms_metabolic.json.gz ]; then
    echo "./data/db/templates/retro.templates.bkms_metabolic.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/templates/retro.templates.bkms_metabolic.json.gz \
      "https://www.dropbox.com/scl/fi/7wvn1d0kp3excrnbvlrm0/retro.templates.bkms_metabolic.json.gz?rlkey=qqzdm8sdrnlaca6ad9jo9aqa3&dl=1"
    echo "retro.templates.bkms_metabolic.json.gz Downloaded."
fi

if [ ! -f ./data/db/templates/retro.templates.cas.json.gz ]; then
    if [ -n "${DROPBOX_LINK_PASSWORD}" ]; then
        echo "./data/db/templates/retro.templates.cas.json.gz not found. Downloading.."
        curl -X POST https://content.dropboxapi.com/2/sharing/get_shared_link_file \
          --header "Authorization: Bearer ${DROPBOX_ACCESS_TOKEN}" \
          --header "Dropbox-API-Arg: {\"path\":\"/retro.templates.cas.json.gz\",\"url\":\"https://www.dropbox.com/scl/fi/o9myvcq06h1mrp3rar5vo/retro.templates.cas.json.gz?rlkey=tw1jt2fqsedfnlh310u1dfwzv&dl=0\", \"link_password\":\"${DROPBOX_LINK_PASSWORD}\"}" \
          -o ./data/db/templates/retro.templates.cas.json.gz
        echo "retro.templates.cas.json.gz Downloaded."
    fi
fi

if [ ! -f ./data/db/templates/retro.templates.pistachio_ringbreaker.json.gz ]; then
    echo "./data/db/templates/retro.templates.pistachio_ringbreaker.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/templates/retro.templates.pistachio_ringbreaker.json.gz \
      "https://www.dropbox.com/scl/fi/axbxwxa0o9stkvhvjos98/retro.templates.pistachio_ringbreaker.json.gz?rlkey=grzwpo6gkh788d2bu8g8k89an&dl=1"
    echo "retro.templates.pistachio_ringbreaker.json.gz Downloaded."
fi

if [ ! -f ./data/db/templates/retro.templates.pistachio.json.gz ]; then
    echo "./data/db/templates/retro.templates.pistachio.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/templates/retro.templates.pistachio.json.gz \
      "https://www.dropbox.com/scl/fi/rbws4la330itj1mp69pmn/retro.templates.pistachio.json.gz?rlkey=8dcgei1h1f2k534fiyw92062e&dl=1"
    echo "retro.templates.pistachio.json.gz Downloaded."
fi

if [ ! -f ./data/db/templates/retro.templates.reaxys_biocatalysis.json.gz ]; then
    echo "./data/db/templates/retro.templates.reaxys_biocatalysis.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/templates/retro.templates.reaxys_biocatalysis.json.gz \
      "https://www.dropbox.com/scl/fi/yf97rfpxul9sxb6r1bqdb/retro.templates.reaxys_biocatalysis.json.gz?rlkey=kj11a60gu3zch2b802bgzq0ui&dl=1"
    echo "retro.templates.reaxys_biocatalysis.json.gz Downloaded."
fi

if [ ! -f ./data/db/templates/retro.templates.reaxys.json.gz ]; then
    echo "./data/db/templates/retro.templates.reaxys.json.gz not found. Downloading.."
    wget -q --show-progress -O data/db/templates/retro.templates.reaxys.json.gz \
      "https://www.dropbox.com/scl/fi/31hc3r6w1atwqqrh2mfm1/retro.templates.reaxys.json.gz?rlkey=w8dgkympjlcll51299k6nbv12&dl=1"
    echo "retro.templates.reaxys.json.gz Downloaded."
fi
