def target_date_allocation(assets, goal_period):
    # 주식 및 채권 비율 설정 _ 아주 긴 장기적투자라고 생각하고 임의로 비율 넣음 
    stock_weight = min(0.9, max(0.1, goal_period / 20))  
    bond_weight = 1 - stock_weight 

    stock_assets = [asset for asset in assets if "stock" in asset.lower() or "spy" in asset.lower()]
    bond_assets = [asset for asset in assets if "bond" in asset.lower() or "govt" in asset.lower()]

    if stock_assets and bond_assets:
        stock_alloc = {asset: stock_weight / len(stock_assets) for asset in stock_assets}
        bond_alloc = {asset: bond_weight / len(bond_assets) for asset in bond_assets}
        return {**stock_alloc, **bond_alloc}
    elif stock_assets:
        return {asset: 1.0 / len(stock_assets) for asset in stock_assets}
    elif bond_assets:
        return {asset: 1.0 / len(bond_assets) for asset in bond_assets}
    else:
        raise ValueError("No stock or bond assets found in the provided asset list.")

